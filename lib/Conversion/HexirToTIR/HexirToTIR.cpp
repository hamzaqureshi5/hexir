//===- LowerToTIR.cpp - hexir graph ops to hextir kernels -----------------===//
//
// Lowers the graph level to the kernel level: each compute op in the `hexir`
// dialect becomes a `hextir.prim_func` at module scope plus a
// `hexir.call_tir` at the use site.
//
// This is where placement stops being an annotation and becomes code. The
// `device` attribute PartitionPass put on the op decides the *kind* of the
// parallel loops in the generated kernel:
//
//   device = "cpu"   ->  hextir.for "parallel"
//   device = "cuda"  ->  hextir.for "thread_binding" bind "blockIdx.x" / ...
//
// so the schedule is visible in the IR rather than implied by which pass runs
// later. Reduction axes stay "serial" on both.
//
// `hexir.constant` and `hexir.print` are deliberately left alone -- they are
// not compute and have no kernel.
//
//===----------------------------------------------------------------------===//

#include "hexir/Dialect/Hexir/IR/HexirDialect.h"
#include "hexir/Dialect/HexTIR/IR/HexTIRDialect.h"
#include "hexir/Conversion/Passes.h"
#include "hexir/Dialect/Hexir/Transforms/Passes.h"
#include "hexir/Dialect/LS/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

using namespace mlir;

namespace {

/// The device an op was placed on. Unannotated ops default to CPU, matching
/// TargetSupport's fallback.
static StringRef deviceOf(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>("device"))
    return attr.getValue();
  return "cpu";
}

/// Tensor type -> the buffer type a prim func takes for it.
static MemRefType bufferTypeFor(Type type) {
  auto tensorTy = cast<RankedTensorType>(type);
  return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
}

/// Create one loop level. `parallel` axes take their kind from the placement;
/// reduction axes are always serial. `depth` picks the GPU axis to bind to.
static hextir::ForOp makeLoop(OpBuilder &b, Location loc, Value lb, Value ub,
                              Value step, StringRef device, unsigned depth,
                              bool parallel) {
  StringRef kind = "serial";
  if (parallel)
    kind = device == "cuda" ? "thread_binding" : "parallel";

  auto loop = hextir::ForOp::create(b, loc, lb, ub, step, kind);

  if (parallel && device == "cuda") {
    // Outermost parallel axis maps to blocks, the rest to threads.
    StringRef axis = depth == 0 ? "blockIdx.x" : "threadIdx.x";
    loop->setAttr("thread", b.getStringAttr(axis));
  }
  return loop;
}

//===----------------------------------------------------------------------===//
// Kernel bodies
//===----------------------------------------------------------------------===//

/// C[i][j] = sum_k A[i][k] * B[k][j]
///
/// i and j are parallel, k is the reduction and stays serial.
static void buildMatmulBody(OpBuilder &b, Location loc, hextir::PrimFuncOp fn,
                            StringRef device) {
  Block &entry = fn.getBody().front();
  Value A = entry.getArgument(0);
  Value B = entry.getArgument(1);
  Value C = entry.getArgument(2);

  auto aTy = cast<MemRefType>(A.getType());
  auto cTy = cast<MemRefType>(C.getType());
  Type elemTy = cTy.getElementType();

  b.setInsertionPointToStart(&entry);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value cM = arith::ConstantIndexOp::create(b, loc, cTy.getShape()[0]);
  Value cN = arith::ConstantIndexOp::create(b, loc, cTy.getShape()[1]);
  Value cK = arith::ConstantIndexOp::create(b, loc, aTy.getShape()[1]);
  Value zero = arith::ConstantOp::create(b, loc, b.getZeroAttr(elemTy));

  auto blockOp = hextir::BlockOp::create(b, loc, b.getStringAttr("matmul"));
  Block &blockBody = blockOp.getBody().emplaceBlock();
  // Terminate the block region up front and build before the terminator, the
  // same shape hextir.for's builder uses.
  b.setInsertionPointToEnd(&blockBody);
  auto blockTerm = hextir::YieldOp::create(b, loc, ValueRange{});
  b.setInsertionPoint(blockTerm);

  auto iLoop = makeLoop(b, loc, c0, cM, c1, device, /*depth=*/0, true);
  b.setInsertionPointToStart(&iLoop.getBody().front());

  auto jLoop = makeLoop(b, loc, c0, cN, c1, device, /*depth=*/1, true);
  b.setInsertionPointToStart(&jLoop.getBody().front());

  Value i = iLoop.getInductionVar();
  Value j = jLoop.getInductionVar();

  // Zero the accumulator before the reduction.
  hextir::BufferStoreOp::create(b, loc, zero, C, ValueRange{i, j});

  auto kLoop = makeLoop(b, loc, c0, cK, c1, device, /*depth=*/2, false);
  b.setInsertionPointToStart(&kLoop.getBody().front());
  Value k = kLoop.getInductionVar();

  Value a = hextir::BufferLoadOp::create(b, loc, elemTy, A, ValueRange{i, k});
  Value bv = hextir::BufferLoadOp::create(b, loc, elemTy, B, ValueRange{k, j});
  Value acc = hextir::BufferLoadOp::create(b, loc, elemTy, C, ValueRange{i, j});
  Value prod = arith::MulFOp::create(b, loc, a, bv);
  Value sum = arith::AddFOp::create(b, loc, acc, prod);
  hextir::BufferStoreOp::create(b, loc, sum, C, ValueRange{i, j});

  b.setInsertionPointToEnd(&entry);
  hextir::ReturnOp::create(b, loc, ValueRange{});
}

/// Element-wise kernel over every axis. `emit` builds the scalar body from the
/// loaded input values and returns the value to store.
static void buildElementwiseBody(
    OpBuilder &b, Location loc, hextir::PrimFuncOp fn, StringRef device,
    unsigned numInputs, StringRef blockName,
    llvm::function_ref<Value(OpBuilder &, Location, ValueRange)> emit) {
  Block &entry = fn.getBody().front();
  Value dst = entry.getArgument(numInputs);
  auto dstTy = cast<MemRefType>(dst.getType());
  Type elemTy = dstTy.getElementType();

  b.setInsertionPointToStart(&entry);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);

  SmallVector<Value> extents;
  for (int64_t dim : dstTy.getShape())
    extents.push_back(arith::ConstantIndexOp::create(b, loc, dim));

  auto blockOp = hextir::BlockOp::create(b, loc, b.getStringAttr(blockName));
  Block &blockBody = blockOp.getBody().emplaceBlock();
  b.setInsertionPointToEnd(&blockBody);
  auto blockTerm = hextir::YieldOp::create(b, loc, ValueRange{});
  b.setInsertionPoint(blockTerm);

  // Every axis of an element-wise op is parallel.
  SmallVector<Value> ivs;
  for (auto [depth, extent] : llvm::enumerate(extents)) {
    auto loop = makeLoop(b, loc, c0, extent, c1, device, depth, true);
    b.setInsertionPointToStart(&loop.getBody().front());
    ivs.push_back(loop.getInductionVar());
  }

  SmallVector<Value> loaded;
  for (unsigned idx = 0; idx < numInputs; ++idx)
    loaded.push_back(hextir::BufferLoadOp::create(
        b, loc, elemTy, entry.getArgument(idx), ivs));

  Value result = emit(b, loc, loaded);
  hextir::BufferStoreOp::create(b, loc, result, dst, ivs);

  b.setInsertionPointToEnd(&entry);
  hextir::ReturnOp::create(b, loc, ValueRange{});
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerToTIRPass
    : public PassWrapper<LowerToTIRPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToTIRPass)

  StringRef getArgument() const final { return "hexir-lower-to-tir"; }
  StringRef getDescription() const final {
    return "Lower hexir compute ops to hextir prim funcs called via call_tir";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, hextir::HexTIRDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    unsigned counter = 0;

    // Collect first: the walk creates new module-level symbols as it goes.
    SmallVector<Operation *> work;
    module.walk([&](Operation *op) {
      if (isa<hexir::LinearOp, hexir::AddOp, hexir::ReluOp>(op))
        work.push_back(op);
    });

    for (Operation *op : work)
      if (failed(lowerOne(module, op, counter)))
        return signalPassFailure();
  }

  LogicalResult lowerOne(ModuleOp module, Operation *op, unsigned &counter) {
    Location loc = op->getLoc();
    StringRef device = deviceOf(op);

    if (!isa<RankedTensorType>(op->getResult(0).getType()))
      return op->emitError("expected a ranked tensor result");

    // One buffer per operand, plus one for the result: the destination-passing
    // contract hexir.call_tir's verifier enforces.
    SmallVector<Type> bufferTypes;
    for (Value operand : op->getOperands())
      bufferTypes.push_back(bufferTypeFor(operand.getType()));
    bufferTypes.push_back(bufferTypeFor(op->getResult(0).getType()));

    std::string name =
        (llvm::Twine(op->getName().stripDialect()) + "_" + llvm::Twine(counter++))
            .str();

    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    auto fn = hextir::PrimFuncOp::create(
        builder, loc, name,
        builder.getFunctionType(bufferTypes, /*results=*/{}));
    fn->setAttr("device", builder.getStringAttr(device));

    unsigned numInputs = op->getNumOperands();
    if (isa<hexir::LinearOp>(op)) {
      buildMatmulBody(builder, loc, fn, device);
    } else if (isa<hexir::AddOp>(op)) {
      buildElementwiseBody(
          builder, loc, fn, device, numInputs, "add",
          [](OpBuilder &b, Location l, ValueRange in) -> Value {
            return arith::AddFOp::create(b, l, in[0], in[1]);
          });
    } else {
      // relu: max(x, 0)
      buildElementwiseBody(
          builder, loc, fn, device, numInputs, "relu",
          [](OpBuilder &b, Location l, ValueRange in) -> Value {
            Value zero =
                arith::ConstantOp::create(b, l, b.getZeroAttr(in[0].getType()));
            return arith::MaximumFOp::create(b, l, in[0], zero);
          });
    }

    // Replace the graph op with a call to the kernel we just built.
    OpBuilder callBuilder(op);
    auto call = hexir::CallTIROp::create(
        callBuilder, loc, op->getResult(0).getType(),
        FlatSymbolRefAttr::get(module.getContext(), name), op->getOperands());
    call->setAttr("device", callBuilder.getStringAttr(device));

    op->getResult(0).replaceAllUsesWith(call.getResult());
    op->erase();
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hexir::createLowerToTIRPass() {
  return std::make_unique<LowerToTIRPass>();
}
