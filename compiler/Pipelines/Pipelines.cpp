//===- Pipelines.cpp - Hexir compilation pipeline -------------------------===//
//
// Single source of truth for pass ordering. Everything here only *adds* passes
// -- the driver runs the PassManager once, after this returns.
//
// One function per phase, so this file stays readable as passes accumulate.
// The phase graph, and which gating is legitimate where, is documented on
// `Phase` in include/hexir/Pipelines/Pipelines.h.
//
//===----------------------------------------------------------------------===//

#include "hexir/Pipelines/Pipelines.h"

#include "hexir/Conversion/Passes.h"
#include "hexir/Target/CudaToolkit.h"
#include "hexir/Dialect/Hexir/IR/HexirDialect.h"
#include "hexir/Dialect/Hexir/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

bool mlir::hexir::isLinalgRoute(Phase phase) {
  return phase == Phase::Linalg || phase == Phase::LinalgGpu ||
         phase == Phase::LinalgLLVM;
}

namespace {

//===----------------------------------------------------------------------===//
// graph -- hexir to hexir
//===----------------------------------------------------------------------===//
/// Graph-level cleanup, and where graph optimization will go: folding,
/// const-eval and the fusion decision all belong here, on whole tensors,
/// before placement.
///
/// Nested on hexir::FuncOp, which nothing currently builds -- every program is
/// a func.func, so these passes match no operation and do nothing. That bug is
/// milestone M0 in docs/optimizations/plan.md; it is left alone here so this
/// restructuring does not change any output.
void buildGraphPipeline(PassManager &pm) {
  OpPassManager &optPM = pm.nest<mlir::hexir::FuncOp>();
  optPM.addPass(createCanonicalizerPass());
  optPM.addPass(mlir::hexir::createShapeInferencePass());
  optPM.addPass(createCanonicalizerPass());
  optPM.addPass(createCSEPass());
}

//===----------------------------------------------------------------------===//
// kernel -- hexir to hextir (route A)
//===----------------------------------------------------------------------===//
/// Each hexir.linear/add/relu becomes a hextir.prim_func at module scope plus a
/// hexir.call_tir at the use site. Placement already decided in `place` picks
/// the loop kinds.
///
/// `artifact` stops here too: ModuleSerializer builds the command list from
/// kernel-level IR, and compiles device code on a clone so this module stays at
/// this level.
void buildKernelPipeline(PassManager &pm) {
  pm.addPass(mlir::hexir::createLowerToTIRPass());
}

//===----------------------------------------------------------------------===//
// linalg, linalg-gpu, linalg-llvm -- route B, frozen
//===----------------------------------------------------------------------===//
/// hexir -> linalg -> bufferize -> loops -> LLVM, with cuda-placed linalg ops
/// going through gpu.launch on the way.
///
/// Nothing executes this route: it lost its only consumer when the JIT was
/// removed, so all three phases terminate in a dump. It also bypasses the
/// kernel level, which means no scheduling work applies to it. Frozen until
/// host code generation from hextir replaces it -- see
/// docs/optimizations/plan.md.
void buildLinalgRoutePipeline(PassManager &pm,
                              const mlir::hexir::PipelineOptions &opts) {
  const mlir::hexir::Phase phase = opts.phase;
  // Within this route the phases really are ordered, so `>=` is honest here.
  const bool loweringCudaToGpu = phase >= mlir::hexir::Phase::LinalgGpu;
  const bool loweringToLLVM = phase >= mlir::hexir::Phase::LinalgLLVM;

  // HexirToLinalg propagates the `device` attr onto the linalg ops it creates.
  pm.addPass(mlir::hexir::createLowerToLinalgPass());

  // Second run: fallback for linalg ops that did not inherit a device attr
  // (PartitionPass skips ops already annotated).
  pm.addPass(mlir::hexir::createPartitionPass());

  // A detour through mirror ls_cpu/ls_gpu ops used to exist purely to show
  // placement in the op name; it was a lossy round trip -- rebuilding each op
  // dropped its outs operand and every attribute but `device` -- and printing
  // the real linalg op says strictly more.
  if (phase == mlir::hexir::Phase::Linalg)
    return;

  //===--------------------------------------------------------------------===//
  // Tensor -> MemRef
  //===--------------------------------------------------------------------===//
  // tensor.empty has no bufferization of its own and must become an
  // alloc_tensor first, or the alloc_tensor created mid-flight never enters
  // the bufferization worklist and survives as an unbufferized op.
  pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());

  bufferization::OneShotBufferizePassOptions bufferizeOpts;
  // Keep allowUnknownOps=false: every tensor op must have a
  // BufferizableOpInterface (hexir.print gets one from
  // compiler/Support/BufferizableOpInterfaceImpl.cpp), so a genuine gap is reported
  // as a failure rather than silently leaving tensors in the IR.
  // IdentityLayoutMap keeps materialized buffers as plain memrefs, which is
  // what hexir.print's F64MemRef constraint and the printf lowering expect.
  bufferizeOpts.unknownTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOpts));
  // Fold the to_tensor/to_buffer pairs the materialization leaves behind.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(bufferization::createBufferDeallocationSimplificationPass());

  if (loweringCudaToGpu)
    pm.addPass(mlir::hexir::createCudaGpuLoweringPass());
  if (phase == mlir::hexir::Phase::LinalgGpu)
    return;

  //===--------------------------------------------------------------------===//
  // Linalg -> loops, then finish the GPU path
  //===--------------------------------------------------------------------===//
  // Must run before GPU lowering completes: gpu-to-llvm cannot handle live
  // linalg ops.
  pm.addPass(createConvertLinalgToLoopsPass());

  // Outline the kernel first, then lower structured control flow across the
  // whole module so the kernel body -- which CudaToGpu builds out of scf.for --
  // is flat before anything tries to translate it to LLVM IR.
  if (loweringCudaToGpu)
    pm.addPass(createGpuKernelOutliningPass());

  // SCF -> CFG, module-wide: host and kernel both need it.
  pm.addPass(createSCFToControlFlowPass());

  if (loweringCudaToGpu) {
    // Order matches MLIR's own gpu-lower-to-nvvm-pipeline, and it is not
    // interchangeable:
    //
    //   nvvm-attach-target   attach the NVPTX target to the gpu.module
    //   convert-gpu-to-nvvm  gpu/arith/func -> NVVM, inside the gpu.module
    //   gpu-to-llvm          host side: launch_func -> CUDA runtime calls
    //   gpu-module-to-binary NVVM -> PTX -> CUBIN, needs ptxas
    //
    // gpu-to-llvm must come BEFORE gpu-module-to-binary. Run the other way
    // round and the binary is already embedded when the host conversion runs,
    // which fails with "failed to legalize operation 'gpu.launch_func'".
    GpuNVVMAttachTargetOptions nvvmOpts;
    nvvmOpts.chip = opts.gpuChip;
    nvvmOpts.features = opts.gpuFeatures;
    nvvmOpts.optLevel = opts.gpuOptLevel;
    pm.addPass(createGpuNVVMAttachTarget(nvvmOpts));

    {
      OpPassManager &gpuPM = pm.nest<gpu::GPUModuleOp>();
      gpuPM.addPass(createConvertGpuOpsToNVVMOps());
      // A partial conversion, so it leaves unrealized_conversion_casts behind.
      // The reconcile pass at the end of the pipeline only sees the host
      // module, so the kernel needs its own or gpu-module-to-binary fails to
      // translate it.
      gpuPM.addPass(createReconcileUnrealizedCastsPass());
    }

    // Host side to LLVM first, then the GPU handoff -- again matching
    // gpu-lower-to-nvvm-pipeline, which converts func/arith/memref before
    // gpu-to-llvm. Leave gpu.launch_func standing for the next pass.
    pm.addPass(mlir::hexir::createLowerToLLVMPass());
    pm.addPass(createGpuToLLVMConversionPass());
    GpuModuleToBinaryPassOptions binaryOpts;
    // Ubuntu does not lay the toolkit out the way MLIR expects; see
    // compiler/Target/CudaToolkit.cpp.
    binaryOpts.toolkitPath = mlir::hexir::resolveCudaToolkitPath();
    pm.addPass(createGpuModuleToBinaryPass(binaryOpts));
  }

  //===--------------------------------------------------------------------===//
  // LLVM dialect
  //===--------------------------------------------------------------------===//
  if (loweringToLLVM) {
    // Already run above on the CUDA path, before the GPU handoff.
    if (!loweringCudaToGpu)
      pm.addPass(mlir::hexir::createLowerToLLVMPass());
    // Clean up unrealized casts left by the partial conversions above.
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(LLVM::createDIScopeForLLVMFuncOpPass());
  }
}

} // namespace

void mlir::hexir::buildHexirPipeline(PassManager &pm,
                                     const PipelineOptions &opts) {
  const Phase phase = opts.phase;

  // `input` is the graph as it arrived, so it runs nothing -- unless -opt was
  // asked for, which is the one case where a plain dump is expected to have
  // been cleaned up.
  if (phase != Phase::Input || opts.enableOpt)
    buildGraphPipeline(pm);
  if (phase == Phase::Input || phase == Phase::Graph)
    return;

  //===--------------------------------------------------------------------===//
  // place
  //===--------------------------------------------------------------------===//
  // Placement is decided on the frontend hexir ops (hexir.linear, hexir.relu,
  // ...) BEFORE any lowering, and every later phase reads the attribute it
  // leaves behind. It is not optional: a dump without it would show something
  // the compiler never actually works with.
  pm.addPass(mlir::hexir::createPartitionPass());
  if (phase == Phase::Place)
    return;

  //===--------------------------------------------------------------------===//
  // the fork
  //===--------------------------------------------------------------------===//
  if (isLinalgRoute(phase)) {
    buildLinalgRoutePipeline(pm, opts);
    return;
  }

  buildKernelPipeline(pm);
}
