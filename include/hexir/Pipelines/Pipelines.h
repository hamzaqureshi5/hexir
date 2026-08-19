//===- Pipelines.h - Hexir compilation pipeline ----------------*- C++ -*-===//
//
// The pass pipeline, separated from the driver that parses flags and runs it.
//
//===----------------------------------------------------------------------===//

#ifndef HEXIR_PIPELINES_PIPELINES_H
#define HEXIR_PIPELINES_PIPELINES_H

#include <string>

namespace mlir {
class PassManager;

namespace hexir {

/// How far to compile.
///
/// **The order is load-bearing.** The pipeline gates work with `>=` on this
/// enum, so moving an entry changes which passes run. `TIR` sits below
/// `Linalg` on purpose: it is a branch off the pipeline, not a step along it.
enum class Stage {
  None,
  AST,          // -emit=ast          (stub)
  Hexir,        // -emit=mlir
  TIR,          // -emit=mlir-tir     terminal; nothing lowers hextir further
  HXB,          // -emit=hxb          serialize the TIR-level module to a file
  Affine,       // -emit=mlir-affine  (no affine pass is wired up)
  Linalg,       // -emit=mlir-linalg
  Hetero,       // -emit=mlir-hetero
  GPU,          // -emit=mlir-gpu
  LLVMDialect,  // -emit=mlir-llvm
  LLVMIR,       // -emit=llvm
  JIT           // -emit=jit
};

struct PipelineOptions {
  Stage stage = Stage::JIT;
  bool enableOpt = false;

  /// NVPTX target for the CUDA path. sm_86 is the A6000; use sm_80 for A100,
  /// sm_89 for L4/4090, sm_75 for a 1660 Ti.
  std::string gpuChip = "sm_75";
  std::string gpuFeatures = "+ptx80";
  int gpuOptLevel = 3;
};

/// Add every pass needed to reach `opts.stage`, and nothing past it, so the
/// caller runs the PassManager exactly once.
void buildHexirPipeline(PassManager &pm, const PipelineOptions &opts);

/// True while the result is still MLIR, rather than LLVM IR or a JIT run.
bool stageEmitsMLIR(Stage stage);

} // namespace hexir
} // namespace mlir

#endif // HEXIR_PIPELINES_PIPELINES_H
