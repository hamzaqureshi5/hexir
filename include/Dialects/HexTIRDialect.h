//===- HexTIRDialect.h - Hexir kernel-level dialect ------------*- C++ -*-===//
//
// See HexTIROps.td for what this level is for.
//
//===----------------------------------------------------------------------===//

#ifndef HEXTIR_DIALECT_H
#define HEXTIR_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialects/HexTIRDialect.h.inc"

#define GET_OP_CLASSES
#include "Dialects/HexTIROps.h.inc"

#endif // HEXTIR_DIALECT_H
