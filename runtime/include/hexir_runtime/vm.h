//===- vm.h - Execute a loaded module ----------------------------*- C -*-===//
#ifndef HEXIR_RUNTIME_VM_H
#define HEXIR_RUNTIME_VM_H

#include "hexir_runtime/hal.h"
#include "hexir_runtime/module.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Run the named entry point of `module` on `device`.
hexir_status_t hexir_execute(const hexir_module_t *module,
                             hexir_device_t *device, const char *entry);

/// Run, and copy the destination of the last dispatch into `out_result`.
///
/// Exists so a caller can check what the program actually computed. A
/// benchmark that times a kernel without verifying its output will happily
/// report a great number for a kernel that computes a fraction of the answer.
hexir_status_t hexir_execute_capture(const hexir_module_t *module,
                                     hexir_device_t *device, const char *entry,
                                     void *out_result, size_t out_size);

#ifdef __cplusplus
}
#endif
#endif // HEXIR_RUNTIME_VM_H
