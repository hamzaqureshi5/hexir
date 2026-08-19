//===- hal_internal.h - Backend vtable ---------------------------*- C -*-===//
//
// Backends implement this and nothing else. Adding CUDA means adding one
// vtable in src/hal/cuda/, not touching any caller.
//
//===----------------------------------------------------------------------===//
#ifndef HEXIR_RUNTIME_HAL_INTERNAL_H
#define HEXIR_RUNTIME_HAL_INTERNAL_H

#include "hexir_runtime/hal.h"

typedef struct hexir_device_vtable_t {
  const char *name;
  hexir_status_t (*buffer_allocate)(hexir_device_t *, size_t,
                                    hexir_memory_kind_t, hexir_buffer_t **);
  void (*buffer_release)(hexir_buffer_t *);
  hexir_status_t (*buffer_write)(hexir_buffer_t *, const void *, size_t);
  hexir_status_t (*buffer_read)(const hexir_buffer_t *, void *, size_t);
  void *(*buffer_host_pointer)(hexir_buffer_t *);
  hexir_status_t (*wait)(hexir_device_t *);
  void (*release)(hexir_device_t *);
} hexir_device_vtable_t;

struct hexir_device_t {
  const hexir_device_vtable_t *vtable;
  hexir_device_kind_t kind;
  void *impl;
  /* Optional, more specific than vtable->name: the CUDA backend fills in the
     actual GPU so output says which card ran the work. */
  char name[128];
};

struct hexir_buffer_t {
  hexir_device_t *device;
  size_t size;
  hexir_memory_kind_t memory;
  void *impl;
};

hexir_status_t hexir_cpu_device_create(hexir_device_t **out_device);
hexir_status_t hexir_cuda_device_create(hexir_device_t **out_device);

#endif // HEXIR_RUNTIME_HAL_INTERNAL_H
