//===- runtime.h - Umbrella header -------------------------------*- C -*-===//
//
// The hexir runtime: loads a compiled module and runs it. Deliberately has no
// dependency on MLIR or LLVM -- that is the whole point of shipping an
// artifact rather than a JIT.
//
//===----------------------------------------------------------------------===//
#ifndef HEXIR_RUNTIME_RUNTIME_H
#define HEXIR_RUNTIME_RUNTIME_H

#include "hexir_runtime/hal.h"
#include "hexir_runtime/module.h"
#include "hexir_runtime/status.h"
#include "hexir_runtime/vm.h"

#endif // HEXIR_RUNTIME_RUNTIME_H
