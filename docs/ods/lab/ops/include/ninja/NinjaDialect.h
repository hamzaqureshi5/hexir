//===- NinjaDialect.h - A small dialect for learning ODS -------*- C++ -*-===//
//
// See NinjaOps.td.
//
//===----------------------------------------------------------------------===//

#ifndef NINJA_DIALECT_H
#define NINJA_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ninja/NinjaDialect.h.inc"
#include "ninja/NinjaEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ninja/NinjaTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ninja/NinjaAttrs.h.inc"

#define GET_OP_CLASSES
#include "ninja/NinjaOps.h.inc"

#endif // NINJA_DIALECT_H
