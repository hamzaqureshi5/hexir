//===- LSDialects.h - Local-system CPU/GPU dialects ------------*- C++ -*-===//

#ifndef LS_DIALECTS_H
#define LS_DIALECTS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "hexir/Dialect/LS/IR/LSCPUDialect.h.inc"
#include "hexir/Dialect/LS/IR/LSGPUDialect.h.inc"

#define GET_OP_CLASSES
#include "hexir/Dialect/LS/IR/LSDialectsOps.h.inc"

#endif // LS_DIALECTS_H
