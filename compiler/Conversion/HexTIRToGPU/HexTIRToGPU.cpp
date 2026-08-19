//===- HexTIRToGPU.cpp - hextir kernels to the GPU dialect ----------------===//
//
// Turns a `hextir.prim_func` placed on cuda into a `gpu.module` holding a
// `gpu.func`, which is the form MLIR can compile to NVVM and then to a CUBIN.
//
// This is where the loop `kind` finally pays off. A "thread_binding" loop does
// not become a loop at all: its induction variable is read from the hardware,
// and the extent becomes a launch dimension. Everything else stays a loop.
//
//   hextir.for "thread_binding" %c0 to %cN step %c1 bind "blockIdx.x"
//     ->  %i = gpu.block_id x        (grid.x = N)
//
//   hextir.for "thread_binding" ... bind "threadIdx.x"
//     ->  %j = gpu.thread_id x       (block.x = N)
//
//   hextir.for "serial" / "parallel"  ->  scf.for
//
// Only the shapes LowerToTIR emits are handled: rank-2 buffers, at most one
// bound block axis and one bound thread axis. Anything else is reported rather
// than mis-compiled.
//
//===----------------------------------------------------------------------===//

#include "hexir/Conversion/Passes.h"
#include "hexir/Dialect/HexTIR/IR/HexTIRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

using namespace mlir;

namespace {

/// Constant extent of a loop whose bounds LowerToTIR emitted as constants.
static std::optional<int64_t> constantExtent(Value lower, Value upper) {
  APInt lo, hi;
  if (!matchPattern(lower, m_ConstantInt(&lo)) ||
      !matchPattern(upper, m_ConstantInt(&hi)))
    return std::nullopt;
  return hi.getSExtValue() - lo.getSExtValue();
}

struct HexTIRToGPUPass
    : public PassWrapper<HexTIRToGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HexTIRToGPUPass)

  StringRef getArgument() const final { return "hextir-to-gpu"; }
  StringRef getDescription() const final {
    return "Lower cuda-placed hextir prim funcs into gpu.module kernels";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<hextir::PrimFuncOp> kernels;
    for (auto fn : module.getOps<hextir::PrimFuncOp>()) {
      auto device = fn->getAttrOfType<StringAttr>("device");
      if (device && device.getValue() == "cuda")
        kernels.push_back(fn);
    }

    for (hextir::PrimFuncOp fn : kernels)
      if (failed(convert(module, fn)))
        return signalPassFailure();
  }

  LogicalResult convert(ModuleOp module, hextir::PrimFuncOp fn) {
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());

    // One gpu.module per kernel keeps the mapping to an executable entry
    // one-to-one, which is what the artifact format wants.
    auto gpuModule = gpu::GPUModuleOp::create(
        builder, fn.getLoc(), (fn.getSymName() + "_module").str());

    builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
    auto gpuFunc = gpu::GPUFuncOp::create(builder, fn.getLoc(), fn.getSymName(),
                                          fn.getFunctionType());
    // The inherent attribute, not a discardable "gpu.kernel". Setting the
    // discardable one as well makes GPUFuncOpLowering merge both into the
    // llvm.func and trip an assertion about duplicate attribute names.
    gpuFunc.setKernelAttr(builder.getUnitAttr());

    IRMapping mapping;
    Block &src = fn.getBody().front();
    Block &dst = gpuFunc.getBody().front();
    for (auto [from, to] : llvm::zip(src.getArguments(), dst.getArguments()))
      mapping.map(from, to);

    builder.setInsertionPointToStart(&dst);
    if (failed(cloneBody(builder, src, mapping, fn)))
      return failure();

    builder.setInsertionPointToEnd(&dst);
    gpu::ReturnOp::create(builder, fn.getLoc());

    // The prim func stays. Erasing it would leave the hexir.call_tir that
    // names it dangling, and the caller of this pass (the serializer, working
    // on a throwaway clone) only wants the compiled kernel out of it.
    fn->setAttr("hexir.lowered_to_gpu", builder.getUnitAttr());
    return success();
  }

  /// Clone `block` into the builder's insertion point, rewriting hextir ops.
  LogicalResult cloneBody(OpBuilder &builder, Block &block, IRMapping &mapping,
                          hextir::PrimFuncOp fn) {
    for (Operation &op : block) {
      if (isa<hextir::YieldOp, hextir::ReturnOp>(op))
        continue;

      if (auto blockOp = dyn_cast<hextir::BlockOp>(op)) {
        // A schedulable region is only a name at this point; inline it.
        if (failed(cloneBody(builder, blockOp.getBody().front(), mapping, fn)))
          return failure();
        continue;
      }

      if (auto forOp = dyn_cast<hextir::ForOp>(op)) {
        if (failed(cloneLoop(builder, forOp, mapping, fn)))
          return failure();
        continue;
      }

      if (auto load = dyn_cast<hextir::BufferLoadOp>(op)) {
        Value value = memref::LoadOp::create(
            builder, load.getLoc(), mapping.lookup(load.getBuffer()),
            llvm::to_vector(llvm::map_range(load.getIndices(), [&](Value v) {
              return mapping.lookup(v);
            })));
        mapping.map(load.getResult(), value);
        continue;
      }

      if (auto store = dyn_cast<hextir::BufferStoreOp>(op)) {
        memref::StoreOp::create(
            builder, store.getLoc(), mapping.lookup(store.getValue()),
            mapping.lookup(store.getBuffer()),
            llvm::to_vector(llvm::map_range(store.getIndices(), [&](Value v) {
              return mapping.lookup(v);
            })));
        continue;
      }

      // arith and anything else clones unchanged.
      builder.clone(op, mapping);
    }
    return success();
  }

  LogicalResult cloneLoop(OpBuilder &builder, hextir::ForOp forOp,
                          IRMapping &mapping, hextir::PrimFuncOp fn) {
    Location loc = forOp.getLoc();
    StringRef kind = forOp.getKind();

    if (kind == "thread_binding") {
      auto thread = forOp.getThread();
      if (!thread)
        return forOp.emitOpError("thread_binding loop has no bound axis");

      // The induction variable comes from the hardware; the extent becomes a
      // launch dimension, recorded on the func for the host to read back.
      Value id;
      if (thread->starts_with("blockIdx"))
        id = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::x);
      else if (thread->starts_with("threadIdx"))
        id = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
      else
        return forOp.emitOpError("unsupported bound axis '") << *thread << "'";

      std::optional<int64_t> extent =
          constantExtent(forOp.getLowerBound(), forOp.getUpperBound());
      if (!extent)
        return forOp.emitOpError("thread_binding loop needs a constant extent");
      // Recorded on the enclosing gpu.module: the func's attributes are
      // rewritten when it is lowered, the module's are not.
      auto gpuModule = builder.getInsertionBlock()
                           ->getParentOp()
                           ->getParentOfType<gpu::GPUModuleOp>();
      gpuModule->setAttr(thread->starts_with("blockIdx") ? "hexir.grid_x"
                                                         : "hexir.block_x",
                         builder.getI64IntegerAttr(*extent));

      mapping.map(forOp.getBody().front().getArgument(0), id);
      return cloneBody(builder, forOp.getBody().front(), mapping, fn);
    }

    // serial and parallel both stay loops inside the kernel.
    auto loop = scf::ForOp::create(
        builder, loc, mapping.lookupOrDefault(forOp.getLowerBound()),
        mapping.lookupOrDefault(forOp.getUpperBound()),
        mapping.lookupOrDefault(forOp.getStep()));
    mapping.map(forOp.getBody().front().getArgument(0), loop.getInductionVar());

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    return cloneBody(builder, forOp.getBody().front(), mapping, fn);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hexir::createHexTIRToGPUPass() {
  return std::make_unique<HexTIRToGPUPass>();
}
