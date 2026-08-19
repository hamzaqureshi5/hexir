//===- hal.h - Hardware abstraction layer ------------------------*- C -*-===//
//
// The boundary between the host program and whatever actually runs the
// kernels. Everything device-specific sits behind this vtable, so the host
// program in a module is identical whether it ends up on CPU or CUDA.
//
// Only the CPU backend is implemented. The CUDA backend goes in
// src/hal/cuda/ and must talk to the driver API (libcuda) rather than the
// CUDA runtime API, so it can be dlopened on machines that have no toolkit.
//
//===----------------------------------------------------------------------===//
#ifndef HEXIR_RUNTIME_HAL_H
#define HEXIR_RUNTIME_HAL_H

#include <stddef.h>
#include "hexir_runtime/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  HEXIR_DEVICE_CPU = 0,
  HEXIR_DEVICE_CUDA = 1,
} hexir_device_kind_t;

typedef struct hexir_device_t hexir_device_t;
typedef struct hexir_buffer_t hexir_buffer_t;

/// Where a buffer lives, and therefore what it costs to touch it from the
/// host. Placement decisions in the compiler show up here as the difference
/// between DEVICE_LOCAL and HOST_VISIBLE.
typedef enum {
  HEXIR_MEMORY_HOST_LOCAL = 0,  /* host RAM */
  HEXIR_MEMORY_DEVICE_LOCAL = 1,/* device RAM, needs an explicit transfer */
  HEXIR_MEMORY_HOST_VISIBLE = 2,/* mapped/unified, no explicit transfer */
} hexir_memory_kind_t;

hexir_status_t hexir_device_create(hexir_device_kind_t kind,
                                   hexir_device_t **out_device);
void hexir_device_release(hexir_device_t *device);
const char *hexir_device_name(const hexir_device_t *device);
hexir_device_kind_t hexir_device_kind(const hexir_device_t *device);

/// Block until previously submitted work on this device has completed. A no-op
/// for the CPU backend, which is synchronous.
hexir_status_t hexir_device_wait(hexir_device_t *device);

hexir_status_t hexir_buffer_allocate(hexir_device_t *device, size_t size,
                                     hexir_memory_kind_t memory,
                                     hexir_buffer_t **out_buffer);
void hexir_buffer_release(hexir_buffer_t *buffer);
size_t hexir_buffer_size(const hexir_buffer_t *buffer);
hexir_memory_kind_t hexir_buffer_memory_kind(const hexir_buffer_t *buffer);

/// NULL when the buffer is DEVICE_LOCAL: use write/read instead.
void *hexir_buffer_host_pointer(hexir_buffer_t *buffer);

hexir_status_t hexir_buffer_write(hexir_buffer_t *dst, const void *src,
                                  size_t size);
hexir_status_t hexir_buffer_read(const hexir_buffer_t *src, void *dst,
                                 size_t size);

#ifdef __cplusplus
}
#endif
#endif // HEXIR_RUNTIME_HAL_H
