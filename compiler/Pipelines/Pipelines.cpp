//===- Pipelines.cpp - Hexir compilation pipeline -------------------------===//
//
// Single source of truth for pass ordering. Everything here only *adds* passes
// -- the driver runs the PassManager once, after this returns.
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

bool mlir::hexir::stageEmitsMLIR(Stage stage) {
  return stage <= Stage::LLVMDialect;
}

void mlir::hexir::buildHexirPipeline(PassManager &pm,
                                     const PipelineOptions &opts) {
  const Stage stage = opts.stage;

  const bool loweringToLinalg = stage >= Stage::Linalg;
  const bool partitioningForHetero = stage >= Stage::Hetero;
  // GPU lowering runs for every stage at or beyond mlir-gpu (incl. llvm/jit).
  const bool loweringCudaToGpu = stage >= Stage::GPU;
  const bool loweringToLLVM = stage >= Stage::LLVMDialect;

  //===--------------------------------------------------------------------===//
  // Graph-level cleanup
  //===--------------------------------------------------------------------===//
  const bool kernelLevel = stage == Stage::TIR || stage == Stage::HXB;

  if (opts.enableOpt || loweringToLinalg || kernelLevel) {
    // Nested on hexir::FuncOp, which only exists if hexir.func ops were built.
    OpPassManager &optPM = pm.nest<mlir::hexir::FuncOp>();
    optPM.addPass(createCanonicalizerPass());
    optPM.addPass(mlir::hexir::createShapeInferencePass());
    optPM.addPass(createCanonicalizerPass());
    optPM.addPass(createCSEPass());
  }

  //===--------------------------------------------------------------------===//
  // Kernel level (branch)
  //===--------------------------------------------------------------------===//
  // Partition first so every compute op carries a `device`, then turn each one
  // into a hextir.prim_func. Terminal: nothing lowers hextir further yet, so
  // this does not fall through to the linalg pipeline below.
  // -emit=hxb serializes from exactly this level, so it shares the pipeline.
  if (kernelLevel) {
    pm.addPass(mlir::hexir::createPartitionPass());
    pm.addPass(mlir::hexir::createLowerToTIRPass());
    return;
  }

  if (!loweringToLinalg)
    return;

  //===--------------------------------------------------------------------===//
  // Placement, then hexir -> linalg
  //===--------------------------------------------------------------------===//
  // Placement is decided on the frontend hexir ops (hexir.linear, hexir.relu,
  // ...) BEFORE lowering; HexirToLinalg propagates the device attr onto the
  // linalg ops it creates.
  if (partitioningForHetero)
    pm.addPass(mlir::hexir::createPartitionPass());

  pm.addPass(mlir::hexir::createLowerToLinalgPass());
  if (stage == Stage::Linalg)
    return;

  if (partitioningForHetero) {
    // Second run: fallback for linalg ops that did not inherit a device attr
    // (PartitionPass skips ops already annotated).
    pm.addPass(mlir::hexir::createPartitionPass());
  }
  // -emit=mlir-hetero stops here and prints linalg carrying `device`. There
  // used to be a detour through mirror ls_cpu/ls_gpu ops purely to show
  // placement in the op name; it was a lossy round trip -- rebuilding each op
  // dropped its outs operand and every attribute but `device` -- and printing
  // the real linalg op says strictly more.
  if (stage == Stage::Hetero)
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
  if (stage == Stage::GPU)
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
