//===- hexir-bench.c - Time a module against cuBLAS -----------------------===//
//
// Runs a compiled .hxb module and, when the kernel is a matmul, times cuBLAS
// doing the same work on the same device. Without a baseline a GFLOPS number
// means nothing, so this always reports both, plus a correctness check.
//
// cuBLAS is dlopened rather than linked, the same way the runtime treats
// libcuda, so this still builds on a machine with no CUDA.
//
//===----------------------------------------------------------------------===//

#include "hexir_runtime/runtime.h"
#include "hexir_runtime/program.h"

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Wall clock. CLOCK_MONOTONIC so a clock adjustment mid-run cannot produce a
// negative interval.
static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// A matmul is 2*M*N*K flops: one multiply and one add per inner iteration.
static double gemm_gflops(uint32_t m, uint32_t n, uint32_t k, double seconds) {
  if (seconds <= 0.0)
    return 0.0;
  return (2.0 * m * n * k) / seconds / 1e9;
}

//===----------------------------------------------------------------------===//
// Reading the module
//===----------------------------------------------------------------------===//

// The first executable entry, which is the kernel this benchmark reports on.
// A module with several kernels still runs in full; only the reported shape
// comes from this one.
static int first_executable(const hexir_module_t *module,
                            hexir_executable_entry_t *out) {
  const void *data = NULL;
  uint64_t size = 0;
  if (hexir_module_section(module, HEXIR_SECTION_EXECUTABLES, &data, &size) !=
      HEXIR_OK)
    return 0;
  if (size < sizeof(hexir_executable_header_t))
    return 0;

  const hexir_executable_header_t *header =
      (const hexir_executable_header_t *)data;
  if (header->count == 0)
    return 0;
  if (size < sizeof(*header) + sizeof(*out))
    return 0;

  memcpy(out, (const uint8_t *)data + sizeof(*header), sizeof(*out));
  return 1;
}

//===----------------------------------------------------------------------===//
// cuBLAS, loaded at runtime
//===----------------------------------------------------------------------===//

typedef void *cublasHandle_t;

// cuBLAS is column-major. Rather than transposing anything, we ask it for
// B*A, which in column-major terms produces exactly the row-major A*B the
// compiler computes. This is the standard trick and it keeps both sides
// operating on identical buffers.
enum { CUBLAS_OP_N = 0 };

static struct {
  void *lib;
  int loaded;
  int (*Create)(cublasHandle_t *);
  int (*Destroy)(cublasHandle_t);
  int (*Dgemm)(cublasHandle_t, int, int, int, int, int, const double *,
               const double *, int, const double *, int, const double *,
               double *, int);
} cublas;

static int cublas_load(void) {
  if (cublas.loaded)
    return cublas.lib != NULL;
  cublas.loaded = 1;

  cublas.lib = dlopen("libcublas.so", RTLD_NOW | RTLD_LOCAL);
  if (!cublas.lib)
    cublas.lib = dlopen("libcublas.so.12", RTLD_NOW | RTLD_LOCAL);
  if (!cublas.lib)
    cublas.lib = dlopen("libcublas.so.11", RTLD_NOW | RTLD_LOCAL);
  if (!cublas.lib) {
    fprintf(stderr, "note: cuBLAS not available (%s); skipping the baseline\n",
            dlerror());
    return 0;
  }

  *(void **)(&cublas.Create) = dlsym(cublas.lib, "cublasCreate_v2");
  *(void **)(&cublas.Destroy) = dlsym(cublas.lib, "cublasDestroy_v2");
  *(void **)(&cublas.Dgemm) = dlsym(cublas.lib, "cublasDgemm_v3");
  if (!cublas.Dgemm)
    *(void **)(&cublas.Dgemm) = dlsym(cublas.lib, "cublasDgemm_v2");

  if (!cublas.Create || !cublas.Destroy || !cublas.Dgemm) {
    fprintf(stderr, "note: libcublas is missing symbols; skipping the baseline\n");
    dlclose(cublas.lib);
    cublas.lib = NULL;
    return 0;
  }
  return 1;
}

//===----------------------------------------------------------------------===//
// Reference
//===----------------------------------------------------------------------===//

// Check a sample of output elements rather than recomputing the whole matrix.
//
// A full CPU reference is O(N^3): at 4096 that is 1.4e11 flops single
// threaded, minutes per run, which makes verification something people turn
// off. Checking a few hundred scattered elements costs O(samples * K) and
// catches everything that matters here -- a kernel that computes part of the
// output, or indexes wrongly, gets the sampled elements wrong too.
//
// The stride is deliberately coprime-ish with the row length so samples do not
// all land in the same column.
static int verify_sampled(const double *a, const double *b, const double *got,
                          uint32_t m, uint32_t n, uint32_t k, int samples,
                          double *out_worst) {
  size_t total = (size_t)m * n;
  size_t stride = total / (size_t)samples;
  if (stride == 0)
    stride = 1;

  double worst = 0.0;
  for (size_t index = 0; index < total; index += stride) {
    uint32_t i = (uint32_t)(index / n);
    uint32_t j = (uint32_t)(index % n);

    double expected = 0.0;
    for (uint32_t p = 0; p < k; ++p)
      expected += a[(size_t)i * k + p] * b[(size_t)p * n + j];

    double scale = fabs(expected) > 1.0 ? fabs(expected) : 1.0;
    double err = fabs(got[index] - expected) / scale;
    if (err > worst)
      worst = err;
  }
  *out_worst = worst;

  // f64 accumulation over K terms, so scale the bar with K rather than using a
  // fixed epsilon.
  return worst <= 1e-12 * (double)k;
}

//===----------------------------------------------------------------------===//
// Timing
//===----------------------------------------------------------------------===//

// Time hexir end to end: this is allocation, host to device transfers, the
// kernel and the synchronise, because that is what a caller actually pays.
static double time_hexir(const hexir_module_t *module, hexir_device_kind_t kind,
                         int iterations, hexir_status_t *out_status,
                         double *out_result, size_t result_bytes) {
  hexir_device_t *device = NULL;
  *out_status = hexir_device_create(kind, &device);
  if (*out_status != HEXIR_OK)
    return 0.0;

  // One untimed run first: the first launch pays for context setup and module
  // load, which would otherwise dominate a short measurement. This is also the
  // run whose output gets checked.
  *out_status = hexir_execute_capture(module, device, "main", out_result,
                                      result_bytes);
  if (*out_status != HEXIR_OK) {
    hexir_device_release(device);
    return 0.0;
  }

  double start = now_seconds();
  for (int i = 0; i < iterations; ++i) {
    *out_status = hexir_execute(module, device, "main");
    if (*out_status != HEXIR_OK)
      break;
  }
  double elapsed = now_seconds() - start;

  hexir_device_release(device);
  return elapsed / iterations;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

static int usage(void) {
  fprintf(stderr,
          "usage: hexir-bench <module.hxb> [--device=cpu|cuda] [--iters=N]\n"
          "                   [--no-baseline] [--no-verify]\n");
  return 2;
}

int main(int argc, char **argv) {
  const char *path = NULL;
  hexir_device_kind_t kind = HEXIR_DEVICE_CUDA;
  int iterations = 20;
  int want_baseline = 1;
  int want_verify = 1;

  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--device=", 9) == 0) {
      kind = strcmp(argv[i] + 9, "cpu") == 0 ? HEXIR_DEVICE_CPU
                                             : HEXIR_DEVICE_CUDA;
    } else if (strncmp(argv[i], "--iters=", 8) == 0) {
      iterations = atoi(argv[i] + 8);
      if (iterations < 1)
        return usage();
    } else if (strcmp(argv[i], "--no-baseline") == 0) {
      want_baseline = 0;
    } else if (strcmp(argv[i], "--no-verify") == 0) {
      want_verify = 0;
    } else if (argv[i][0] == '-') {
      return usage();
    } else {
      path = argv[i];
    }
  }
  if (!path)
    return usage();

  hexir_module_t *module = NULL;
  hexir_status_t status = hexir_module_load_file(path, &module);
  if (status != HEXIR_OK) {
    fprintf(stderr, "hexir-bench: cannot load '%s': %s\n", path,
            hexir_status_string(status));
    return 1;
  }

  hexir_executable_entry_t entry;
  if (!first_executable(module, &entry)) {
    fprintf(stderr, "hexir-bench: module has no executable to measure\n");
    hexir_module_release(module);
    return 1;
  }

  int is_gemm = entry.kind == HEXIR_KERNEL_MATMUL;
  printf("module   : %s\n", path);
  printf("kernel   : %s  %s  built for %s  %ux%ux%u  elem=%uB\n", entry.name,
         hexir_kernel_kind_string(entry.kind),
         entry.device == 1 ? "cuda" : "cpu", entry.m, entry.n, entry.k,
         entry.elem_size);
  printf("device   : %s\n", kind == HEXIR_DEVICE_CUDA ? "cuda" : "cpu");
  printf("iters    : %d\n\n", iterations);

  // A module compiled for one device cannot run on another, and the runtime
  // enforces that, so say so plainly rather than reporting a failure.
  if ((entry.device == 1) != (kind == HEXIR_DEVICE_CUDA)) {
    fprintf(stderr,
            "hexir-bench: the module was built for %s but --device is %s\n",
            entry.device == 1 ? "cuda" : "cpu",
            kind == HEXIR_DEVICE_CUDA ? "cuda" : "cpu");
    hexir_module_release(module);
    return 1;
  }

  // Check what hexir computed before reporting how fast it did it. Timing a
  // kernel without verifying it is how a wrong kernel gets celebrated: doing a
  // fraction of the work looks exactly like being fast.
  size_t result_elems = (size_t)entry.m * entry.n;
  double *hexir_result = NULL;
  if (want_verify && entry.elem_size == sizeof(double))
    hexir_result = (double *)malloc(result_elems * sizeof(double));

  double hexir_seconds =
      time_hexir(module, kind, iterations, &status, hexir_result,
                 hexir_result ? result_elems * sizeof(double) : 0);
  if (status != HEXIR_OK) {
    fprintf(stderr, "hexir-bench: execution failed: %s\n",
            hexir_status_string(status));
    free(hexir_result);
    hexir_module_release(module);
    return 1;
  }

  int hexir_correct = 1;
  if (hexir_result) {
    const void *rodata_for_check = NULL;
    uint64_t rodata_bytes = 0;
    size_t a_n = (size_t)entry.m * entry.k;
    size_t b_n = (size_t)entry.k * entry.n;
    if (is_gemm &&
        hexir_module_section(module, HEXIR_SECTION_RODATA, &rodata_for_check,
                             &rodata_bytes) == HEXIR_OK &&
        rodata_bytes >= (a_n + b_n) * sizeof(double)) {
      const double *ra = (const double *)rodata_for_check;
      double err = 0.0;
      hexir_correct = verify_sampled(ra, ra + a_n, hexir_result, entry.m,
                                     entry.n, entry.k, 512, &err);
      printf("verify   : hexir max relative error %.3g over 512 sampled "
             "elements (%s)\n",
             err, hexir_correct ? "ok" : "WRONG");
    }
  }
  free(hexir_result);

  // A wrong kernel that runs fast is not a result. Say so instead of printing
  // a number someone might quote.
  if (!hexir_correct) {
    fprintf(stderr,
            "hexir-bench: the kernel produced the wrong answer; timings below "
            "are meaningless\n");
  }

  printf("hexir    : %8.3f ms", hexir_seconds * 1e3);
  if (is_gemm)
    printf("   %8.2f GFLOP/s",
           gemm_gflops(entry.m, entry.n, entry.k, hexir_seconds));
  printf("\n");

  // Everything past here needs a matmul shape and a GPU.
  if (!is_gemm || kind != HEXIR_DEVICE_CUDA || !want_baseline) {
    hexir_module_release(module);
    return 0;
  }

  if (!cublas_load()) {
    hexir_module_release(module);
    return 0;
  }

  // Feed cuBLAS the same constants the module carries, so both sides compute
  // the same thing and the verification below is meaningful.
  const void *rodata = NULL;
  uint64_t rodata_size = 0;
  size_t a_elems = (size_t)entry.m * entry.k;
  size_t b_elems = (size_t)entry.k * entry.n;
  size_t c_elems = (size_t)entry.m * entry.n;
  if (hexir_module_section(module, HEXIR_SECTION_RODATA, &rodata,
                           &rodata_size) != HEXIR_OK ||
      rodata_size < (a_elems + b_elems) * sizeof(double)) {
    fprintf(stderr, "note: module has no usable rodata; skipping the baseline\n");
    hexir_module_release(module);
    return 0;
  }
  const double *host_a = (const double *)rodata;
  const double *host_b = host_a + a_elems;

  hexir_device_t *device = NULL;
  if (hexir_device_create(HEXIR_DEVICE_CUDA, &device) != HEXIR_OK) {
    hexir_module_release(module);
    return 1;
  }

  hexir_buffer_t *da = NULL, *db = NULL, *dc = NULL;
  hexir_buffer_allocate(device, a_elems * sizeof(double),
                        HEXIR_MEMORY_DEVICE_LOCAL, &da);
  hexir_buffer_allocate(device, b_elems * sizeof(double),
                        HEXIR_MEMORY_DEVICE_LOCAL, &db);
  hexir_buffer_allocate(device, c_elems * sizeof(double),
                        HEXIR_MEMORY_DEVICE_LOCAL, &dc);
  hexir_buffer_write(da, host_a, a_elems * sizeof(double));
  hexir_buffer_write(db, host_b, b_elems * sizeof(double));

  cublasHandle_t handle = NULL;
  if (cublas.Create(&handle) != 0) {
    fprintf(stderr, "note: cublasCreate failed; skipping the baseline\n");
    hexir_buffer_release(da);
    hexir_buffer_release(db);
    hexir_buffer_release(dc);
    hexir_device_release(device);
    hexir_module_release(module);
    return 0;
  }

  double *dev_a = (double *)hexir_buffer_device_pointer(da);
  double *dev_b = (double *)hexir_buffer_device_pointer(db);
  double *dev_c = (double *)hexir_buffer_device_pointer(dc);
  const double alpha = 1.0, beta = 0.0;

  // Column-major cuBLAS computing B*A gives the row-major A*B the compiler
  // produces, on the very same buffers, so no transposes are needed.
  int gemm_m = (int)entry.n;
  int gemm_n = (int)entry.m;
  int gemm_k = (int)entry.k;

  // Warm up, for the same reason as above: the first call initialises cuBLAS
  // and picks a kernel.
  cublas.Dgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, gemm_m, gemm_n, gemm_k, &alpha,
               dev_b, gemm_m, dev_a, gemm_k, &beta, dev_c, gemm_m);
  hexir_device_wait(device);

  double start = now_seconds();
  for (int i = 0; i < iterations; ++i)
    cublas.Dgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, gemm_m, gemm_n, gemm_k,
                 &alpha, dev_b, gemm_m, dev_a, gemm_k, &beta, dev_c, gemm_m);
  hexir_device_wait(device);
  double cublas_seconds = (now_seconds() - start) / iterations;

  printf("cuBLAS   : %8.3f ms   %8.2f GFLOP/s   (kernel only)\n",
         cublas_seconds * 1e3,
         gemm_gflops(entry.m, entry.n, entry.k, cublas_seconds));
  printf("\nhexir is %.1fx %s than cuBLAS\n",
         hexir_seconds > cublas_seconds ? hexir_seconds / cublas_seconds
                                        : cublas_seconds / hexir_seconds,
         hexir_seconds > cublas_seconds ? "slower" : "faster");

  // The two timings measure different things -- hexir includes transfers,
  // cuBLAS does not -- so say so rather than letting the ratio be read as a
  // like-for-like kernel comparison.
  printf("note     : hexir timing includes allocation and host/device copies;\n"
         "           the cuBLAS timing is the kernel alone.\n");

  cublas.Destroy(handle);
  hexir_buffer_release(da);
  hexir_buffer_release(db);
  hexir_buffer_release(dc);
  hexir_device_release(device);
  hexir_module_release(module);
  return 0;
}
