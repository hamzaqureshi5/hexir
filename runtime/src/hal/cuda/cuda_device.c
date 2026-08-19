//===- cuda_device.c - CUDA backend ---------------------------------------===//
//
// Device memory and transfers through the CUDA driver API.
//
// libcuda is loaded with dlopen rather than linked, and the symbols are looked
// up by hand. That keeps the runtime buildable and runnable on a machine with
// no CUDA at all -- asking for a CUDA device there fails with a message
// instead of failing to load the binary. It is also why this file declares the
// handful of driver types it needs instead of including cuda.h: the runtime has
// no CUDA build dependency, only a runtime one.
//
// Note the _v2 suffixes. The driver API versions its symbols, and the
// unsuffixed names either do not exist or are the older 32-bit-pointer forms.
//
//===----------------------------------------------------------------------===//

#include "../hal_internal.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Minimal slice of the driver API. */
typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef unsigned long long CUdeviceptr;
typedef void *CUmodule;
typedef void *CUfunction;

#define CUDA_SUCCESS 0

static struct {
  void *lib;
  int loaded;
  CUresult (*Init)(unsigned int);
  CUresult (*DeviceGet)(CUdevice *, int);
  CUresult (*DeviceGetName)(char *, int, CUdevice);
  CUresult (*CtxCreate)(CUcontext *, unsigned int, CUdevice);
  CUresult (*CtxDestroy)(CUcontext);
  CUresult (*CtxSynchronize)(void);
  CUresult (*MemAlloc)(CUdeviceptr *, size_t);
  CUresult (*MemFree)(CUdeviceptr);
  CUresult (*MemcpyHtoD)(CUdeviceptr, const void *, size_t);
  CUresult (*MemcpyDtoH)(void *, CUdeviceptr, size_t);
  CUresult (*GetErrorString)(CUresult, const char **);
  CUresult (*ModuleLoadData)(CUmodule *, const void *);
  CUresult (*ModuleUnload)(CUmodule);
  CUresult (*ModuleGetFunction)(CUfunction *, CUmodule, const char *);
  CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                           unsigned, unsigned, unsigned, void *, void **,
                           void **);
} cu;

typedef struct {
  CUcontext context;
  CUdevice device;
} cuda_device_impl_t;

static const char *cuda_error(CUresult result) {
  const char *message = NULL;
  if (cu.GetErrorString && cu.GetErrorString(result, &message) == CUDA_SUCCESS &&
      message)
    return message;
  return "unknown CUDA error";
}

/* Reports on the first failure and gives up; there is no partial mode. */
static int cuda_load(void) {
  if (cu.loaded)
    return cu.lib != NULL;
  cu.loaded = 1;

  /* libcuda.so.1 is the driver, shipped with the GPU driver rather than the
     toolkit. libcuda.so is the toolkit's development symlink and may be
     absent on a machine that only has the driver installed. */
  cu.lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!cu.lib)
    cu.lib = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
  if (!cu.lib) {
    fprintf(stderr, "hexir: cannot load libcuda: %s\n", dlerror());
    return 0;
  }

#define HEXIR_CU_BIND(field, symbol)                                           \
  do {                                                                         \
    *(void **)(&cu.field) = dlsym(cu.lib, symbol);                             \
    if (!cu.field) {                                                           \
      fprintf(stderr, "hexir: libcuda is missing %s\n", symbol);               \
      dlclose(cu.lib);                                                         \
      cu.lib = NULL;                                                           \
      return 0;                                                                \
    }                                                                          \
  } while (0)

  HEXIR_CU_BIND(Init, "cuInit");
  HEXIR_CU_BIND(DeviceGet, "cuDeviceGet");
  HEXIR_CU_BIND(DeviceGetName, "cuDeviceGetName");
  HEXIR_CU_BIND(CtxCreate, "cuCtxCreate_v2");
  HEXIR_CU_BIND(CtxDestroy, "cuCtxDestroy_v2");
  HEXIR_CU_BIND(CtxSynchronize, "cuCtxSynchronize");
  HEXIR_CU_BIND(MemAlloc, "cuMemAlloc_v2");
  HEXIR_CU_BIND(MemFree, "cuMemFree_v2");
  HEXIR_CU_BIND(MemcpyHtoD, "cuMemcpyHtoD_v2");
  HEXIR_CU_BIND(MemcpyDtoH, "cuMemcpyDtoH_v2");
  HEXIR_CU_BIND(ModuleLoadData, "cuModuleLoadData");
  HEXIR_CU_BIND(ModuleUnload, "cuModuleUnload");
  HEXIR_CU_BIND(ModuleGetFunction, "cuModuleGetFunction");
  HEXIR_CU_BIND(LaunchKernel, "cuLaunchKernel");
#undef HEXIR_CU_BIND

  /* Optional: only used to make error messages readable. */
  *(void **)(&cu.GetErrorString) = dlsym(cu.lib, "cuGetErrorString");
  return 1;
}

static hexir_status_t cuda_buffer_allocate(hexir_device_t *device, size_t size,
                                           hexir_memory_kind_t memory,
                                           hexir_buffer_t **out_buffer) {
  (void)memory; /* everything here is device local */
  hexir_buffer_t *buffer = calloc(1, sizeof(*buffer));
  if (!buffer)
    return HEXIR_ERROR_OUT_OF_MEMORY;

  CUdeviceptr pointer = 0;
  if (size) {
    CUresult result = cu.MemAlloc(&pointer, size);
    if (result != CUDA_SUCCESS) {
      fprintf(stderr, "hexir: cuMemAlloc(%zu) failed: %s\n", size,
              cuda_error(result));
      free(buffer);
      return HEXIR_ERROR_OUT_OF_MEMORY;
    }
  }
  buffer->device = device;
  buffer->size = size;
  buffer->memory = HEXIR_MEMORY_DEVICE_LOCAL;
  /* A device address, not something the host may dereference. */
  buffer->impl = (void *)(uintptr_t)pointer;
  *out_buffer = buffer;
  return HEXIR_OK;
}

static void cuda_buffer_release(hexir_buffer_t *buffer) {
  if (!buffer)
    return;
  CUdeviceptr pointer = (CUdeviceptr)(uintptr_t)buffer->impl;
  if (pointer)
    cu.MemFree(pointer);
  free(buffer);
}

static hexir_status_t cuda_buffer_write(hexir_buffer_t *dst, const void *src,
                                        size_t size) {
  CUresult result =
      cu.MemcpyHtoD((CUdeviceptr)(uintptr_t)dst->impl, src, size);
  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "hexir: host to device copy failed: %s\n",
            cuda_error(result));
    return HEXIR_ERROR_DEVICE;
  }
  return HEXIR_OK;
}

static hexir_status_t cuda_buffer_read(const hexir_buffer_t *src, void *dst,
                                       size_t size) {
  CUresult result =
      cu.MemcpyDtoH(dst, (CUdeviceptr)(uintptr_t)src->impl, size);
  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "hexir: device to host copy failed: %s\n",
            cuda_error(result));
    return HEXIR_ERROR_DEVICE;
  }
  return HEXIR_OK;
}

/* Device memory is not host addressable, so callers must use write/read. */
static void *cuda_buffer_host_pointer(hexir_buffer_t *buffer) {
  (void)buffer;
  return NULL;
}

/* Loads the image, finds the kernel and launches it.
 *
 * The module is loaded per dispatch rather than cached. That is a real cost
 * and an obvious thing to fix, but correctness first: caching needs a key that
 * survives the module outliving any one command. */
static hexir_status_t cuda_launch(hexir_device_t *device, const void *image,
                                  size_t image_size, const char *entry,
                                  unsigned grid_x, unsigned grid_y,
                                  unsigned block_x, unsigned block_y,
                                  hexir_buffer_t **args, unsigned arg_count) {
  (void)device;
  (void)image_size; /* the image is self describing */

  CUmodule module = NULL;
  CUresult result = cu.ModuleLoadData(&module, image);
  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "hexir: cannot load device image: %s\n",
            cuda_error(result));
    return HEXIR_ERROR_DEVICE;
  }

  CUfunction function = NULL;
  result = cu.ModuleGetFunction(&function, module, entry);
  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "hexir: no kernel '%s' in the device image: %s\n", entry,
            cuda_error(result));
    cu.ModuleUnload(module);
    return HEXIR_ERROR_NOT_FOUND;
  }

  /* Bare pointer convention: one device address per buffer, and cuLaunchKernel
     wants pointers TO the argument values, not the values. */
  enum { HEXIR_MAX_KERNEL_ARGS = 8 };
  if (arg_count > HEXIR_MAX_KERNEL_ARGS) {
    cu.ModuleUnload(module);
    return HEXIR_ERROR_INVALID_MODULE;
  }
  CUdeviceptr addresses[HEXIR_MAX_KERNEL_ARGS];
  void *params[HEXIR_MAX_KERNEL_ARGS];
  for (unsigned i = 0; i < arg_count; ++i) {
    addresses[i] = (CUdeviceptr)(uintptr_t)args[i]->impl;
    params[i] = &addresses[i];
  }

  result = cu.LaunchKernel(function, grid_x, grid_y, 1, block_x, block_y, 1, 0,
                           NULL, params, NULL);
  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "hexir: launch of '%s' failed: %s\n", entry,
            cuda_error(result));
    cu.ModuleUnload(module);
    return HEXIR_ERROR_DEVICE;
  }

  result = cu.CtxSynchronize();
  cu.ModuleUnload(module);
  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "hexir: kernel '%s' faulted: %s\n", entry,
            cuda_error(result));
    return HEXIR_ERROR_DEVICE;
  }
  return HEXIR_OK;
}

static hexir_status_t cuda_wait(hexir_device_t *device) {
  (void)device;
  CUresult result = cu.CtxSynchronize();
  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "hexir: device synchronize failed: %s\n",
            cuda_error(result));
    return HEXIR_ERROR_DEVICE;
  }
  return HEXIR_OK;
}

static void cuda_release(hexir_device_t *device) {
  if (!device)
    return;
  cuda_device_impl_t *impl = (cuda_device_impl_t *)device->impl;
  if (impl) {
    if (impl->context)
      cu.CtxDestroy(impl->context);
    free(impl);
  }
  free(device);
}

static const hexir_device_vtable_t kCudaVtable = {
    "cuda",
    cuda_buffer_allocate,
    cuda_buffer_release,
    cuda_buffer_write,
    cuda_buffer_read,
    cuda_buffer_host_pointer,
    cuda_launch,
    cuda_wait,
    cuda_release,
};

hexir_status_t hexir_cuda_device_create(hexir_device_t **out_device) {
  if (!cuda_load())
    return HEXIR_ERROR_DEVICE;

  CUresult result = cu.Init(0);
  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "hexir: cuInit failed: %s\n", cuda_error(result));
    return HEXIR_ERROR_DEVICE;
  }

  cuda_device_impl_t *impl = calloc(1, sizeof(*impl));
  hexir_device_t *device = calloc(1, sizeof(*device));
  if (!impl || !device) {
    free(impl);
    free(device);
    return HEXIR_ERROR_OUT_OF_MEMORY;
  }

  result = cu.DeviceGet(&impl->device, 0);
  if (result == CUDA_SUCCESS)
    result = cu.CtxCreate(&impl->context, 0, impl->device);
  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "hexir: cannot open CUDA device 0: %s\n",
            cuda_error(result));
    free(impl);
    free(device);
    return HEXIR_ERROR_DEVICE;
  }

  device->vtable = &kCudaVtable;
  device->kind = HEXIR_DEVICE_CUDA;
  device->impl = impl;

  /* Report the actual GPU rather than just "cuda". */
  char name[96] = {0};
  if (cu.DeviceGetName(name, (int)sizeof(name) - 1, impl->device) ==
      CUDA_SUCCESS)
    snprintf(device->name, sizeof(device->name), "cuda (%s)", name);
  else
    snprintf(device->name, sizeof(device->name), "cuda");

  *out_device = device;
  return HEXIR_OK;
}
