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

#ifdef __cplusplus
}
#endif
#endif // HEXIR_RUNTIME_VM_H
