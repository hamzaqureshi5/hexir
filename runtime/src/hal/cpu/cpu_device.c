//===- cpu_device.c - CPU backend -----------------------------------------===//
//
// Host memory and memcpy. Every buffer is host visible, so write/read are
// copies within one address space and wait() is a no-op -- the CPU backend is
// synchronous by construction.
//
//===----------------------------------------------------------------------===//

#include "../hal_internal.h"

#include <stdlib.h>
#include <string.h>

static hexir_status_t cpu_buffer_allocate(hexir_device_t *device, size_t size,
                                          hexir_memory_kind_t memory,
                                          hexir_buffer_t **out_buffer) {
  (void)memory; /* there is only one kind of memory here */
  hexir_buffer_t *buffer = calloc(1, sizeof(*buffer));
  if (!buffer)
    return HEXIR_ERROR_OUT_OF_MEMORY;

  void *data = size ? calloc(1, size) : NULL;
  if (size && !data) {
    free(buffer);
    return HEXIR_ERROR_OUT_OF_MEMORY;
  }
  buffer->device = device;
  buffer->size = size;
  buffer->memory = HEXIR_MEMORY_HOST_VISIBLE;
  buffer->impl = data;
  *out_buffer = buffer;
  return HEXIR_OK;
}

static void cpu_buffer_release(hexir_buffer_t *buffer) {
  if (!buffer)
    return;
  free(buffer->impl);
  free(buffer);
}

static hexir_status_t cpu_buffer_write(hexir_buffer_t *dst, const void *src,
                                       size_t size) {
  memcpy(dst->impl, src, size);
  return HEXIR_OK;
}

static hexir_status_t cpu_buffer_read(const hexir_buffer_t *src, void *dst,
                                      size_t size) {
  memcpy(dst, src->impl, size);
  return HEXIR_OK;
}

static void *cpu_buffer_host_pointer(hexir_buffer_t *buffer) {
  return buffer->impl;
}

static hexir_status_t cpu_wait(hexir_device_t *device) {
  (void)device;
  return HEXIR_OK;
}

static void cpu_release(hexir_device_t *device) { free(device); }

static const hexir_device_vtable_t kCpuVtable = {
    "cpu",
    cpu_buffer_allocate,
    cpu_buffer_release,
    cpu_buffer_write,
    cpu_buffer_read,
    cpu_buffer_host_pointer,
    cpu_wait,
    cpu_release,
};

hexir_status_t hexir_cpu_device_create(hexir_device_t **out_device) {
  hexir_device_t *device = calloc(1, sizeof(*device));
  if (!device)
    return HEXIR_ERROR_OUT_OF_MEMORY;
  device->vtable = &kCpuVtable;
  device->kind = HEXIR_DEVICE_CPU;
  device->impl = NULL;
  *out_device = device;
  return HEXIR_OK;
}
