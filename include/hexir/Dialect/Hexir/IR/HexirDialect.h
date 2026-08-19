//===- Dialect.h - Dialect definition for the Hexir IR ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Dialect for the Hexir language.
// See docs/Tutorials/Ch-2.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef HEXIR_DIALECT_H_
#define HEXIR_DIALECT_H_

#include "hexir/Dialect/Hexir/IR/ShapeInferenceInterface.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace hexir {

namespace detail {

struct StructTypeStorage;
} // namespace detail
} // namespace hexir
} // namespace mlir

/// Include the auto-generated header file containing the declaration of the hexir
/// dialect.
#include "hexir/Dialect/Hexir/IR/HexirDialect.h.inc"

//===----------------------------------------------------------------------===//
// Hexir Operations
//===----------------------------------------------------------------------===//

/// Include the auto-generated header file containing the declarations of the
/// Hexir operations.
#define GET_OP_CLASSES
#include "hexir/Dialect/Hexir/IR/HexirOps.h.inc"

namespace mlir {
namespace hexir {

//===----------------------------------------------------------------------===//
// Hexir Types
//===----------------------------------------------------------------------===//

/// This class defines the Hexir struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
// class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
//                                                detail::StructTypeStorage> {
// public:
//   /// Inherit some necessary constructors from 'TypeBase'.
//   using Base::Base;

//   /// Create an instance of a `StructType` with the given element types.
//   There
//   /// *must* be atleast one element type.
//   static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

//   /// Returns the element types of this struct type.
//   llvm::ArrayRef<mlir::Type> getElementTypes();

//   /// Returns the number of element type held by this struct.
//   size_t getNumElementTypes() { return getElementTypes().size(); }

//   /// The name of this struct type.
//   static constexpr StringLiteral name = "Hexir.struct";
// };
} // namespace hexir
} // namespace mlir

#endif // HEXIR_DIALECT_H_
