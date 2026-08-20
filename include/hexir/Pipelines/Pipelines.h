//===- Pipelines.h - Hexir compilation phases -----------------*- C++ -*-===//
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
/// **The pipeline is a DAG, not a line, and this enum says where it forks.**
/// The old `Stage` claimed to be a single ordered list and was gated entirely
/// with `>=`, which is how an unimplemented affine stage ended up sitting below
/// the linalg stage and silently running the wrong pipeline.
///
/// What is actually true:
///
///   Input -> Graph -> Place        a shared prefix, ordered, gated with >=
///                       |
///                       +-- Kernel -> Artifact          route A (hextir)
///                       |
///                       +-- Linalg -> LinalgGpu -> LinalgLLVM   route B (frozen)
///
/// So `>=` is valid *within* the prefix and *within* a route, and choosing a
/// route is a branch. A phase from one route says nothing about the other.
///
/// Route B is frozen: nothing has executed it since the JIT was removed, and
/// it bypasses the kernel level, so no scheduling work applies to it. It stays
/// as a dump until host code generation from `hextir` replaces it. See
/// docs/optimizations/plan.md.
///
/// Phases arrive here only once they do something. `schedule`, `host` and
/// `device` are named in the plan and are deliberately absent until the passes
/// behind them exist -- a phase that prints the wrong IR is the bug this enum
/// is being rewritten to prevent.
enum class Phase {
  // Shared prefix.
  Input, // the graph as parsed or built, no passes
  Graph, // after graph optimization
  Place, // after partitioning: `device` on every op, `hexir.targets` on the module

  // Route A: the kernel level, and the only route that produces something
  // runnable.
  Kernel,   // hextir.prim_func per compute op, called via hexir.call_tir
  Artifact, // the .hxb container, written by ModuleSerializer

  // Route B, frozen.
  Linalg,     // linalg on tensors, carrying placement
  LinalgGpu,  // + bufferization, cuda ops as gpu.launch
  LinalgLLVM, // + loops, NVVM, the LLVM dialect
};

/// True for the phases that lower through linalg rather than through hextir.
bool isLinalgRoute(Phase phase);

struct PipelineOptions {
  Phase phase = Phase::Artifact;
  bool enableOpt = false;

  /// NVPTX target for the CUDA path. sm_86 is the A6000; use sm_80 for A100,
  /// sm_89 for L4/4090, sm_75 for a 1660 Ti.
  std::string gpuChip = "sm_75";
  std::string gpuFeatures = "+ptx80";
  int gpuOptLevel = 3;
};

/// Add every pass needed to reach `opts.phase`, and nothing past it, so the
/// caller runs the PassManager exactly once.
void buildHexirPipeline(PassManager &pm, const PipelineOptions &opts);

} // namespace hexir
} // namespace mlir

#endif // HEXIR_PIPELINES_PIPELINES_H
