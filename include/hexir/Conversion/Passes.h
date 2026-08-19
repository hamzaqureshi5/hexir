//===- Passes.h - Hexir conversions ----------------------------*- C++ -*-===//
//
// Passes that lower one dialect into another. Each has an implementation
// directory under compiler/Conversion named for the pair it converts.
//
//===----------------------------------------------------------------------===//

#ifndef HEXIR_CONVERSION_PASSES_H
#define HEXIR_CONVERSION_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace hexir {

/// hexir -> linalg/arith/tensor. Copies the `device` attr onto every op it
/// creates, so placement survives the lowering.
std::unique_ptr<Pass> createLowerToLinalgPass();

/// hexir -> hextir. Each compute op becomes a hextir.prim_func plus a
/// hexir.call_tir; the `device` attr picks the loop kinds.
std::unique_ptr<Pass> createLowerToTIRPass();

/// hexir -> affine. Built but not wired into any pipeline.
std::unique_ptr<Pass> createLowerToAffinePass();

/// Remaining hexir ops (and hexir.print) -> LLVM dialect.
std::unique_ptr<Pass> createLowerToLLVMPass();

/// ls_cpu/ls_gpu -> linalg, preserving `device`.
std::unique_ptr<Pass> createLSTargetsToLinalgPass();

/// linalg ops placed on "cuda" -> gpu.launch.
std::unique_ptr<Pass> createCudaGpuLoweringPass();

} // namespace hexir
} // namespace mlir

#endif // HEXIR_CONVERSION_PASSES_H
