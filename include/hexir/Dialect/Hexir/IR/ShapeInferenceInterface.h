//===- ShapeInferenceInterface.h - Interface definitions for ShapeInference -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef HEXIR_SHAPEINFERENCEINTERFACE_H_
#define HEXIR_SHAPEINFERENCEINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace hexir {

/// Include the auto-generated declarations.
#include "hexir/Dialect/Hexir/IR/ShapeInferenceOpInterfaces.h.inc"

} // namespace hexir
} // namespace mlir

#endif // HEXIR_SHAPEINFERENCEINTERFACE_H_
