
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
    cl::values(clEnumValN(Stage::LLVMIR, "llvm", "output the LLVM IR dump")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

// Resolved once in main from -target / -gpu-chip, then read by the pipeline
// and the serializer.
static std::string resolvedGpuChip;

static cl::list<std::string> targetSpecs(
    "target", cl::CommaSeparated, cl::ZeroOrMore,
    cl::desc("Device to compile for, with its architecture: "
             "<cpu|cuda>[:<sm_XX>]. Repeatable. Default cpu and cuda:sm_75."),
    cl::value_desc("device[:arch]"));

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
//   ./hexir -emit=hxb -placement=hexir.linear=cuda
//   ./hexir -emit=mlir-hetero -placement=hexir.linear=cpu,hexir.relu=cuda
static cl::list<std::string> placementOverrides(
    "placement", cl::CommaSeparated, cl::ZeroOrMore,
    cl::desc("Override op placement: <op-name>=<cpu|cuda>[,...]"),
    cl::value_desc("op=device"));

// What to print when the command line does not make sense.
//
// LLVM's own --help is a wall of options inherited from every linked library,
// so it is close to useless for finding out how to drive this program. This
// lists only what belongs to hexir, and names the valid values rather than
// just saying the input was invalid: someone who mistyped an op needs to see
// the list of ops.
static void printUsage() {
  auto &targets = mlir::hexir::TargetSupport::getInstance();

  llvm::errs() << R"(usage: hexir [options] [input.mlir]

Compiles a program in the hexir dialect. With no input file it compiles the
program built in C++ by compiler/Support/Builder.cpp.

  -emit=<stage>          what to produce (required to get output)
       mlir              the graph as written
       mlir-tir          each compute op as a hextir kernel
       hxb               a loadable module for the runtime; see -o
       mlir-linalg       after lowering to linalg on tensors
       mlir-hetero       placement, as device attributes on linalg ops
       mlir-gpu          cuda-placed ops as gpu.launch
       mlir-llvm         the LLVM dialect
       llvm              translated LLVM IR

  -o <file>              output path for -emit=hxb (default out.hxb)
  -target=<device>[:<arch>]
                         device to compile for, and which chip. Repeatable.
                         default: cpu, cuda:sm_75
  -placement=<op>=<device>[,...]
                         where an op runs, overriding the default
  -opt                   run LLVM's O3 pipeline

examples:
  hexir -emit=mlir-tir -placement=hexir.linear=cuda
  hexir -emit=hxb -o model.hxb -target=cuda:sm_86 -placement=hexir.linear=cuda
  hexir -emit=mlir-hetero -placement=hexir.linear=cuda,hexir.relu=cpu prog.mlir

placeable ops:
)";

  for (const std::string &op : targets.knownOps()) {
    llvm::errs() << "  " << op << "  ->  ";
    llvm::interleaveComma(targets.targetsFor(op), llvm::errs());
    llvm::errs() << "\n";
  }
  llvm::errs() << "\ndevices: cpu, cuda (gpu is accepted as an alias)\n";
}

// Apply -placement overrides to the TargetSupport registry. Returns failure
// on malformed entries or unsupported op/device combinations.
static llvm::LogicalResult applyPlacementOverrides() {
  auto &targets = mlir::hexir::TargetSupport::getInstance();
  for (const std::string &entry : placementOverrides) {
    auto [opName, device] = llvm::StringRef(entry).split('=');

    if (opName.empty() || device.empty()) {
      llvm::errs() << "error: -placement entry '" << entry
                   << "' is not of the form <op-name>=<device>\n\n";
      return llvm::failure();
    }
    if (device != "cpu" && device != "cuda" && device != "gpu") {
      llvm::errs() << "error: '" << device
                   << "' is not a device; expected cpu, cuda or gpu\n\n";
      return llvm::failure();
    }
    // An unknown op and an op that cannot run on the requested device are
    // different mistakes, so say which one happened.
    if (!targets.isKnownOp(opName)) {
      llvm::errs() << "error: '" << opName
                   << "' is not a placeable op\n\n";
      return llvm::failure();
    }
    if (!targets.setPreferredTarget(opName, device)) {
      llvm::errs() << "error: op '" << opName << "' cannot run on '" << device
                   << "'; it supports ";
      llvm::interleaveComma(targets.targetsFor(opName), llvm::errs());
      llvm::errs() << "\n\n";
      return llvm::failure();
    }
  }
  return llvm::success();
}

// Parse -target=<device>[:<arch>] entries. The architecture belongs with the
// device it describes, not in a separate flag: "compile for cuda" and "compile
// for sm_86" are one decision, and splitting them lets them disagree.
static llvm::LogicalResult applyTargets(std::string &gpuChipOut) {
  for (const std::string &entry : targetSpecs) {
    auto [device, arch] = llvm::StringRef(entry).split(':');
    if (device == "cpu") {
      if (!arch.empty()) {
        llvm::errs() << "error: the cpu target takes no architecture ('"
                     << entry << "')\n\n";
        return llvm::failure();
      }
      continue;
    }
    if (device == "cuda" || device == "gpu") {
      if (!arch.empty()) {
        if (!arch.starts_with("sm_")) {
          llvm::errs() << "error: '" << arch
                       << "' is not a CUDA architecture; expected sm_XX, for "
                          "example sm_75\n\n";
          return llvm::failure();
        }
        gpuChipOut = arch.str();
      }
      continue;
    }
    llvm::errs() << "error: '" << device
                 << "' is not a device; expected cpu or cuda\n\n";
    return llvm::failure();
  }
  return llvm::success();
}

static int loadMLIR(mlir::MLIRContext &context,
                    mlir::OwningOpRef<mlir::ModuleOp> &module) {

  context.getOrLoadDialect<mlir::func::FuncDialect>();
  // Register the MLIR -> LLVM IR translations BEFORE any pass runs.
  // gpu-module-to-binary serializes a gpu.module during the pipeline, and it
  // needs these; registering them later was too late and
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
  opts.gpuChip = resolvedGpuChip;
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

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "hexir compiler\n");

  // Nothing to do without -emit, and silently doing nothing is unhelpful.
  if (emitAction.getNumOccurrences() == 0) {
    llvm::errs() << "error: no -emit=<stage> given, so there is nothing to "
                    "produce\n\n";
    printUsage();
    return 2;
  }

  // Targets first: placement is checked against what the registry allows, and
  // the architecture a device was declared with feeds the GPU pipeline.
  resolvedGpuChip = gpuChip;
  if (llvm::failed(applyTargets(resolvedGpuChip))) {
    printUsage();
    return 2;
  }
  if (llvm::failed(applyPlacementOverrides())) {
    printUsage();
    return 2;
  }

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
    if (mlir::failed(mlir::hexir::serializeToHXB(*module, os, resolvedGpuChip)))
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

  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return -1;
}
