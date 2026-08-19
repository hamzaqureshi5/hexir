//===- hexir-run.c - Load and inspect a compiled module -------------------===//
//
// The deploy-side entry point: takes a module the compiler produced and runs
// it, with no compiler in the process.
//
// Loads a module produced by `hexir -emit=hxb`, reports what is in it, then
// replays its command list against the HAL. --selftest exercises the HAL
// without needing a module.
//
//===----------------------------------------------------------------------===//

#include "hexir_runtime/runtime.h"

#include <stdio.h>
#include <string.h>

static int usage(void) {
  fprintf(stderr,
          "usage: hexir-run <module.hxb> [--device=cpu|cuda] [--entry=name]\n"
          "       hexir-run --selftest\n");
  return 2;
}

/* Exercises the HAL without needing a module: allocate, write, read back. */
static int selftest(hexir_device_kind_t kind) {
  hexir_device_t *device = NULL;
  hexir_status_t status = hexir_device_create(kind, &device);
  if (status != HEXIR_OK) {
    fprintf(stderr, "hexir-run: device create failed: %s\n",
            hexir_status_string(status));
    return 1;
  }
  printf("device        : %s\n", hexir_device_name(device));

  double in[4] = {1.0, 2.0, 3.0, 4.0};
  double out[4] = {0};
  hexir_buffer_t *buffer = NULL;

  status = hexir_buffer_allocate(device, sizeof(in), HEXIR_MEMORY_DEVICE_LOCAL,
                                 &buffer);
  if (status == HEXIR_OK)
    status = hexir_buffer_write(buffer, in, sizeof(in));
  if (status == HEXIR_OK)
    status = hexir_device_wait(device);
  if (status == HEXIR_OK)
    status = hexir_buffer_read(buffer, out, sizeof(out));

  if (status != HEXIR_OK) {
    fprintf(stderr, "hexir-run: hal selftest failed: %s\n",
            hexir_status_string(status));
    hexir_buffer_release(buffer);
    hexir_device_release(device);
    return 1;
  }

  int ok = memcmp(in, out, sizeof(in)) == 0;
  printf("hal roundtrip : %s (%zu bytes)\n", ok ? "ok" : "MISMATCH",
         hexir_buffer_size(buffer));

  hexir_buffer_release(buffer);
  hexir_device_release(device);
  return ok ? 0 : 1;
}

int main(int argc, char **argv) {
  const char *path = NULL;
  const char *entry = "main";
  hexir_device_kind_t kind = HEXIR_DEVICE_CPU;
  int want_selftest = 0;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--selftest") == 0) {
      want_selftest = 1;
    } else if (strncmp(argv[i], "--entry=", 8) == 0) {
      entry = argv[i] + 8;
    } else if (strncmp(argv[i], "--device=", 9) == 0) {
      const char *name = argv[i] + 9;
      if (strcmp(name, "cpu") == 0) {
        kind = HEXIR_DEVICE_CPU;
      } else if (strcmp(name, "cuda") == 0) {
        kind = HEXIR_DEVICE_CUDA;
      } else {
        fprintf(stderr, "hexir-run: unknown device '%s'\n", name);
        return usage();
      }
    } else if (argv[i][0] == '-') {
      return usage();
    } else {
      path = argv[i];
    }
  }

  if (want_selftest)
    return selftest(kind);
  if (!path)
    return usage();

  hexir_module_t *module = NULL;
  hexir_status_t status = hexir_module_load_file(path, &module);
  if (status != HEXIR_OK) {
    fprintf(stderr, "hexir-run: cannot load '%s': %s\n", path,
            hexir_status_string(status));
    return 1;
  }

  printf("module        : %s\n", path);
  printf("version       : %u\n", hexir_module_version(module));
  printf("sections      : %u\n", hexir_module_section_count(module));
  for (uint32_t i = 0; i < hexir_module_section_count(module); ++i) {
    const hexir_section_entry_t *s = hexir_module_section_at(module, i);
    printf("  [%u] %-12s offset=%-10llu size=%llu\n", i,
           hexir_section_kind_string(s->kind),
           (unsigned long long)s->offset, (unsigned long long)s->size);
  }

  hexir_device_t *device = NULL;
  status = hexir_device_create(kind, &device);
  if (status != HEXIR_OK) {
    fprintf(stderr, "hexir-run: device create failed: %s\n",
            hexir_status_string(status));
    hexir_module_release(module);
    return 1;
  }
  printf("device        : %s\n", hexir_device_name(device));
  printf("--\n");

  status = hexir_execute(module, device, entry);
  if (status != HEXIR_OK)
    fprintf(stderr, "hexir-run: %s failed: %s\n", entry,
            hexir_status_string(status));

  hexir_device_release(device);
  hexir_module_release(module);
  return status == HEXIR_OK ? 0 : 1;
}
