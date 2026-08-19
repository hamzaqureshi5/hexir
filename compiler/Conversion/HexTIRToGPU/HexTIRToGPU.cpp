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
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <string>

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
      // A yield carrying values is handled by cloneLoop, which needs it to
      // build the scf.for terminator; a bare one is just a block terminator.
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
      // The suffix picks the hardware axis. Ignoring it and always reading .x
      // silently makes two different loops read the same index, which computes
      // one row of the answer very quickly.
      gpu::Dimension dim;
      if (thread->ends_with(".x"))
        dim = gpu::Dimension::x;
      else if (thread->ends_with(".y"))
        dim = gpu::Dimension::y;
      else if (thread->ends_with(".z"))
        dim = gpu::Dimension::z;
      else
        return forOp.emitOpError("bound axis '")
               << *thread << "' has no .x/.y/.z suffix";

      Value id;
      if (thread->starts_with("blockIdx"))
        id = gpu::BlockIdOp::create(builder, loc, dim);
      else if (thread->starts_with("threadIdx"))
        id = gpu::ThreadIdOp::create(builder, loc, dim);
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
      // One attribute per axis, so two bound loops cannot overwrite each
      // other's extent.
      std::string name = (thread->starts_with("blockIdx") ? "hexir.grid_"
                                                          : "hexir.block_") +
                         thread->substr(thread->size() - 1).str();
      gpuModule->setAttr(name, builder.getI64IntegerAttr(*extent));

      mapping.map(forOp.getBody().front().getArgument(0), id);
      return cloneBody(builder, forOp.getBody().front(), mapping, fn);
    }

    // serial and parallel both stay loops inside the kernel.
    SmallVector<Value> initArgs;
    for (Value init : forOp.getInitArgs())
      initArgs.push_back(mapping.lookupOrDefault(init));

    auto loop = scf::ForOp::create(
        builder, loc, mapping.lookupOrDefault(forOp.getLowerBound()),
        mapping.lookupOrDefault(forOp.getUpperBound()),
        mapping.lookupOrDefault(forOp.getStep()), initArgs);
    mapping.map(forOp.getBody().front().getArgument(0), loop.getInductionVar());
    for (auto [from, to] :
         llvm::zip(forOp.getRegionIterArgs(), loop.getRegionIterArgs()))
      mapping.map(from, to);
    for (auto [from, to] : llvm::zip(forOp.getResults(), loop.getResults()))
      mapping.map(from, to);

    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      if (failed(cloneBody(builder, forOp.getBody().front(), mapping, fn)))
        return failure();

      // scf.for's builder gives an empty-yield terminator when there are no
      // iter args, and none at all when there are. Either way, replace it with
      // a yield of the values hextir.yield carries.
      auto srcYield =
          cast<hextir::YieldOp>(forOp.getBody().front().getTerminator());
      SmallVector<Value> yielded;
      for (Value value : srcYield.getValues())
        yielded.push_back(mapping.lookupOrDefault(value));

      // scf.for's builder only installs a yield when there are no iter args:
      // with them it cannot know what to yield, so there may be no terminator
      // at all. getTerminator() asserts in that case, hence the guard.
      if (loop.getBody()->mightHaveTerminator())
        if (Operation *existing = loop.getBody()->getTerminator())
          existing->erase();
      builder.setInsertionPointToEnd(loop.getBody());
      scf::YieldOp::create(builder, loc, yielded);
    }
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hexir::createHexTIRToGPUPass() {
  return std::make_unique<HexTIRToGPUPass>();
}
