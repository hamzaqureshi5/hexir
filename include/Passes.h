//===- Passes.h - Toy Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Toy.
//
//===----------------------------------------------------------------------===//

#ifndef PASSES_H
#define PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace hexir {
std::unique_ptr<Pass> createShapeInferencePass();

/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> createLowerToAffinePass();

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

// Create pass hexir-to-linalg
std::unique_ptr<mlir::Pass> createLowerToLinalgPass();

// Lower hexir compute ops to hextir prim funcs called via hexir.call_tir.
// Reads the `device` attr set by the partition pass to pick loop kinds.
std::unique_ptr<mlir::Pass> createLowerToTIRPass();

// Annotate supported ops with their selected CPU/CUDA device.
std::unique_ptr<mlir::Pass> createPartitionPass();

// Materialize partition annotations as ls_cpu/ls_gpu model operations.
std::unique_ptr<mlir::Pass> createMaterializeLSTargetsPass();

// Lower CUDA-partitioned linalg ops to the MLIR GPU dialect.
std::unique_ptr<mlir::Pass> createCudaGpuLoweringPass();

// Lower ls_cpu/ls_gpu model ops back to Linalg (runs after materialization,
// before bufferization, so downstream passes see standard linalg ops).
std::unique_ptr<mlir::Pass> createLSTargetsToLinalgPass();

} // namespace hexir
} // namespace mlir

#endif // HEXIR_PASSES_H
