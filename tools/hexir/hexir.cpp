
#include "hexir/Support/BufferizableOpInterfaceImpl.h"
#include "hexir/Support/Builder.h"
#include "hexir/Dialect/Hexir/IR/HexirDialect.h"
#include "hexir/Dialect/HexTIR/IR/HexTIRDialect.h"
#include "hexir/Conversion/Passes.h"
#include "hexir/Pipelines/Pipelines.h"
#include "hexir/Serialization/ModuleSerializer.h"
#include "hexir/Dialect/Hexir/Transforms/Passes.h"
#include "hexir/Target/TargetInfo.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Target/LLVM/NVVM/Target.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

using namespace hexir;
using namespace builder;
namespace cl = llvm::cl;

using namespace hexir;
using namespace builder;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input hexir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { HEXIR, MLIR };
} // namespace
static cl::opt<enum InputType>
    inputType("x", cl::init(HEXIR),
              cl::desc("Decided the kind of output desired"),
              cl::values(clEnumValN(HEXIR, "hexir",
                                    "load the input file as a hexir source.")),
              cl::values(clEnumValN(MLIR, "mlir",
                                    "load the input file as an MLIR file")));

using mlir::hexir::Stage;

static cl::opt<Stage> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(Stage::Hexir, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(Stage::TIR, "mlir-tir",
                          "output the MLIR dump after lowering hexir compute "
                          "ops to hextir prim funcs")),
    cl::values(clEnumValN(Stage::HXB, "hxb",
                          "serialize a loadable module for the runtime "
                          "(see -o)")),
    cl::values(clEnumValN(Stage::Affine, "mlir-affine",
                          "output the MLIR dump after affine lowering")),
    cl::values(clEnumValN(Stage::Linalg, "mlir-linalg",
                          "output the MLIR dump after linalg lowering")),
    cl::values(clEnumValN(Stage::Hetero, "mlir-hetero",
                          "output MLIR after CPU/CUDA partitioning")),
    cl::values(clEnumValN(Stage::GPU, "mlir-gpu",
                          "output MLIR after lowering CUDA partitions to GPU")),
    cl::values(clEnumValN(Stage::LLVMDialect, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(Stage::LLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(
        clEnumValN(Stage::JIT, "jit",
                   "JIT the code and run it by invoking the main function")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

static cl::opt<std::string> gpuChip(
    "gpu-chip",
    cl::desc("NVPTX target architecture for CUDA-placed ops (default sm_75). "
             "sm_75 GTX 16xx/RTX 20xx, sm_80 A100, sm_86 A6000/RTX 30xx, "
             "sm_89 L4/RTX 40xx. A mismatch fails at launch with "
             "CUDA_ERROR_NO_BINARY_FOR_GPU, not at compile time."),
    cl::value_desc("sm_XX"), cl::init("sm_75"));

static cl::opt<std::string>
    outputFilename("o", cl::desc("Output file for -emit=hxb"),
                   cl::value_desc("filename"), cl::init("out.hxb"));

// Override op placement at runtime without recompiling, e.g.:
//   ./hexir -emit=jit -placement=linalg.matmul=cpu
//   ./hexir -emit=mlir-hetero -placement=linalg.matmul=cpu,linalg.generic=cuda
static cl::list<std::string> placementOverrides(
    "placement", cl::CommaSeparated, cl::ZeroOrMore,
    cl::desc("Override op placement: <op-name>=<cpu|cuda>[,...]"),
    cl::value_desc("op=device"));

// Apply -placement overrides to the TargetSupport registry. Returns failure
// on malformed entries or unsupported op/device combinations.
static llvm::LogicalResult applyPlacementOverrides() {
  auto &targets = mlir::hexir::TargetSupport::getInstance();
  for (const std::string &entry : placementOverrides) {
    auto [opName, device] = llvm::StringRef(entry).split('=');
    if (opName.empty() ||
        (device != "cpu" && device != "cuda" && device != "gpu")) {
      llvm::errs() << "error: invalid -placement entry '" << entry
                   << "' (expected <op-name>=<cpu|cuda|gpu>)\n";
      return llvm::failure();
    }
    if (!targets.setPreferredTarget(opName, device)) {
      llvm::errs() << "error: op '" << opName << "' does not support target '"
                   << device << "'\n";
      return llvm::failure();
    }
  }
  return llvm::success();
}

static int loadMLIR(mlir::MLIRContext &context,
                    mlir::OwningOpRef<mlir::ModuleOp> &module) {

  context.getOrLoadDialect<mlir::func::FuncDialect>();
  // Register the MLIR -> LLVM IR translations BEFORE any pass runs.
  // gpu-module-to-binary serializes a gpu.module during the pipeline, and it
  // needs these; registering them inside runJit/dumpLLVMIR was too late and
  // failed with "missing LLVMTranslationDialectInterface ... for op:
  // gpu.module" no matter what the CUDA install looked like.
  mlir::registerBuiltinDialectTranslation(context);
  mlir::registerGPUDialectTranslation(context);
  mlir::registerLLVMDialectTranslation(context);
  mlir::registerNVVMDialectTranslation(context);

  context.getOrLoadDialect<mlir::hexir::HexirDialect>();

  // If an input file was given, compile that. Otherwise fall back to the
  // program built in C++ by Builder.cpp, which is what the emit-stage tests
  // exercise. Without this branch `inputFilename` was parsed but never read,
  // so there was no way to feed the compiler any IR -- including new dialects.
  if (inputFilename != "-") {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "error: could not open " << inputFilename << ": "
                   << ec.message() << "\n";
      return -1;
    }
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
      llvm::errs() << "error: could not parse " << inputFilename << "\n";
      return -1;
    }
    return 0;
  }

  // CREATE MODULE FIRST
  module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  createMLPLinearFunction(context, *module);
  // createMLPAddFunction(context, *module);
  // createMLPReluFunction(context, *module);
  return 0;
}

static int loadAndProcessMLIR(mlir::MLIRContext &context,
                              mlir::OwningOpRef<mlir::ModuleOp> &module) {
  if (int error = loadMLIR(context, module))
    return error;

  mlir::PassManager pm(module.get()->getName());
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  mlir::hexir::PipelineOptions opts;
  opts.stage = emitAction;
  opts.enableOpt = enableOpt;
  opts.gpuChip = gpuChip;
  mlir::hexir::buildHexirPipeline(pm, opts);

  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

static int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a hexir AST when the input is MLIR\n";
    return 5;
  }

  // auto moduleAST = parseInputFile(inputFilename);
  //  if (!moduleAST)
  //    return 1;

  // dump(*moduleAST);
  return 0;
}

static int dumpLLVMIR(mlir::ModuleOp module) {

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create target machine and configure the LLVM Module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

static int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();


  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  // Load MLIR runner utilities (printf support) and CUDA runtime if available.
  // To enable actual GPU execution:
  //   1. Install CUDA toolkit:  sudo apt install nvidia-cuda-toolkit
  //   2. Build MLIR with CUDA:  cmake -DMLIR_ENABLE_CUDA_RUNNER=ON ...
  //      (produces /usr/local/lib/libmlir_cuda_runtime.so)
  // Load MLIR runner utils and the CUDA runtime wrapper.
  // libmlir_cuda_runtime.so provides mgpu* symbols (mgpuStreamCreate,
  // mgpuLaunchKernel, etc.) that the lowered IR calls. It is produced by
  // building MLIR on the server with:
  //   cmake ... -DMLIR_ENABLE_CUDA_RUNNER=ON
  // Always load runner utils; only load CUDA runtime if it exists on this
  // machine.
  // Both runtimes are optional and only present on machines where MLIR was
  // built/installed with them. printf resolves from libc either way, so a
  // missing c_runner_utils is not fatal -- don't hand the path to the engine
  // if it isn't there, or it prints a spurious MemoryBuffer error.
  llvm::SmallVector<llvm::StringRef> sharedLibs;
  constexpr const char *runnerUtils =
      "/usr/local/lib/libmlir_c_runner_utils.so";
  constexpr const char *cudaRuntime = "/usr/local/lib/libmlir_cuda_runtime.so";
  if (llvm::sys::fs::exists(runnerUtils))
    sharedLibs.push_back(runnerUtils);
  if (llvm::sys::fs::exists(cudaRuntime))
    sharedLibs.push_back(cudaRuntime);
  engineOptions.sharedLibPaths = sharedLibs;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "hexir compiler\n");

  // Apply -placement overrides before any pass runs.
  if (llvm::failed(applyPlacementOverrides()))
    return 1;

  if (emitAction == Stage::AST)
    return dumpAST();

  // If we aren't dumping the AST, then we are compiling with/to MLIR.
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::tensor::TensorDialect, mlir::linalg::LinalgDialect,
                  mlir::gpu::GPUDialect, mlir::NVVM::NVVMDialect,
                  mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                  mlir::affine::AffineDialect, mlir::math::MathDialect,
                  mlir::LLVM::LLVMDialect, mlir::cf::ControlFlowDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::hextir::HexTIRDialect>();

  // Register dialect extensions BEFORE constructing the context so they are
  // visible to passes that query ConvertToLLVMPatternInterface.
  mlir::func::registerAllExtensions(registry);
  mlir::registerAllExtensions(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  mlir::NVVM::registerConvertGpuToNVVMInterface(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  // Teaches #gpu.select_object how to translate itself to LLVM IR. This is a
  // separate registration from registerGPUDialectTranslation, and without it
  // translating a gpu.binary trips an assertion inside MLIR:
  //   Assertion `offloadingHandler && "Invalid offloading handler."' failed.
  mlir::gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(registry);

  MLIRContext context(registry);

  // Register bufferizable op interface external models AFTER dialects loaded
  // The bufferization dialect's own ops (notably bufferization.alloc_tensor,
  // which `-empty-tensor-to-alloc-tensor` produces) get their
  // BufferizableOpInterface from an external model too. Without this,
  // alloc_tensor is not bufferizable and One-Shot Bufferize leaves it in the
  // IR, then reports "op was not bufferized".
  mlir::bufferization::registerBufferizableOpInterfaceExternalModels(
      const_cast<mlir::DialectRegistry &>(context.getDialectRegistry()));
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      const_cast<mlir::DialectRegistry &>(context.getDialectRegistry()));
  mlir::arith::registerBufferizableOpInterfaceExternalModels(
      const_cast<mlir::DialectRegistry &>(context.getDialectRegistry()));
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(
      const_cast<mlir::DialectRegistry &>(context.getDialectRegistry()));
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(
      const_cast<mlir::DialectRegistry &>(context.getDialectRegistry()));
  mlir::hexir::registerBufferizableOpInterfaceExternalModels(
      const_cast<mlir::DialectRegistry &>(context.getDialectRegistry()));

  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::hexir::HexirDialect>();
  context.getOrLoadDialect<mlir::hextir::HexTIRDialect>();
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module))
    return error;

  // Serialize a loadable module for the runtime.
  if (emitAction == Stage::HXB) {
    std::error_code ec;
    llvm::raw_fd_ostream os(outputFilename, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "error: cannot open '" << outputFilename
                   << "': " << ec.message() << "\n";
      return 1;
    }
    if (mlir::failed(mlir::hexir::serializeToHXB(*module, os, gpuChip)))
      return 1;
    llvm::errs() << "wrote " << outputFilename << "\n";
    return 0;
  }

  // If we aren't exporting to non-mlir, then we are done.
  bool isOutputingMLIR = mlir::hexir::stageEmitsMLIR(emitAction);
  if (isOutputingMLIR) {
    module->dump();
    return 0;
  }

  // Check to see if we are compiling to LLVM IR.
  if (emitAction == Stage::LLVMIR)
    return dumpLLVMIR(*module);

  // Otherwise, we must be running the jit.
  if (emitAction == Stage::JIT)
    return runJit(*module);

  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return -1;
}
