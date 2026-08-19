//===- hal.c - Device dispatch --------------------------------------------===//

#include "hal_internal.h"

#include <stddef.h>

hexir_status_t hexir_device_create(hexir_device_kind_t kind,
                                   hexir_device_t **out_device) {
  if (!out_device)
    return HEXIR_ERROR_INVALID_ARGUMENT;
  *out_device = NULL;
  switch (kind) {
  case HEXIR_DEVICE_CPU:
    return hexir_cpu_device_create(out_device);
  case HEXIR_DEVICE_CUDA:
    return hexir_cuda_device_create(out_device);
  }
  return HEXIR_ERROR_INVALID_ARGUMENT;
}

void hexir_device_release(hexir_device_t *device) {
  if (device && device->vtable->release)
    device->vtable->release(device);
}

const char *hexir_device_name(const hexir_device_t *device) {
  if (!device)
    return "(null)";
  return device->name[0] ? device->name : device->vtable->name;
}

hexir_device_kind_t hexir_device_kind(const hexir_device_t *device) {
  return device ? device->kind : HEXIR_DEVICE_CPU;
}

hexir_status_t hexir_device_wait(hexir_device_t *device) {
  if (!device)
    return HEXIR_ERROR_INVALID_ARGUMENT;
  return device->vtable->wait ? device->vtable->wait(device) : HEXIR_OK;
}

hexir_status_t hexir_buffer_allocate(hexir_device_t *device, size_t size,
                                     hexir_memory_kind_t memory,
                                     hexir_buffer_t **out_buffer) {
  if (!device || !out_buffer)
    return HEXIR_ERROR_INVALID_ARGUMENT;
  *out_buffer = NULL;
  return device->vtable->buffer_allocate(device, size, memory, out_buffer);
}

void hexir_buffer_release(hexir_buffer_t *buffer) {
  if (buffer && buffer->device)
    buffer->device->vtable->buffer_release(buffer);
}

size_t hexir_buffer_size(const hexir_buffer_t *buffer) {
  return buffer ? buffer->size : 0u;
}

hexir_memory_kind_t hexir_buffer_memory_kind(const hexir_buffer_t *buffer) {
  return buffer ? buffer->memory : HEXIR_MEMORY_HOST_LOCAL;
}

void *hexir_buffer_host_pointer(hexir_buffer_t *buffer) {
  if (!buffer || !buffer->device->vtable->buffer_host_pointer)
    return NULL;
  return buffer->device->vtable->buffer_host_pointer(buffer);
}

hexir_status_t hexir_buffer_write(hexir_buffer_t *dst, const void *src,
                                  size_t size) {
  if (!dst || !src)
    return HEXIR_ERROR_INVALID_ARGUMENT;
  if (size > dst->size)
    return HEXIR_ERROR_INVALID_ARGUMENT;
  return dst->device->vtable->buffer_write(dst, src, size);
}

hexir_status_t hexir_device_launch(hexir_device_t *device, const void *image,
                                   size_t image_size, const char *entry,
                                   unsigned grid_x, unsigned block_x,
                                   hexir_buffer_t **args, unsigned arg_count) {
  if (!device || !image || !entry)
    return HEXIR_ERROR_INVALID_ARGUMENT;
  if (!device->vtable->launch)
    return HEXIR_ERROR_UNIMPLEMENTED;
  return device->vtable->launch(device, image, image_size, entry, grid_x,
                                block_x, args, arg_count);
}

hexir_status_t hexir_buffer_read(const hexir_buffer_t *src, void *dst,
                                 size_t size) {
  if (!src || !dst)
    return HEXIR_ERROR_INVALID_ARGUMENT;
  if (size > src->size)
    return HEXIR_ERROR_INVALID_ARGUMENT;
  return src->device->vtable->buffer_read(src, dst, size);
}
