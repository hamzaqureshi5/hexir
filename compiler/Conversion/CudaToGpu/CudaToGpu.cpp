#include "hexir/Conversion/Passes.h"
#include "hexir/Dialect/Hexir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

struct CudaGpuLoweringPass
    : public PassWrapper<CudaGpuLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CudaGpuLoweringPass)

  StringRef getArgument() const final { return "hexir-lower-cuda-to-gpu"; }

  StringRef getDescription() const final {
    return "Lower CUDA-partitioned linalg ops to the MLIR GPU dialect.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<linalg::LinalgOp> cudaOps;

    // Collect ALL cuda-annotated linalg ops, regardless of kind.
    module.walk([&](linalg::LinalgOp op) {
      auto device = op->getAttrOfType<StringAttr>("device");
      if (device && device.getValue() == "cuda")
        cudaOps.push_back(op);
    });

    for (linalg::LinalgOp op : cudaOps) {
      LogicalResult result =
          llvm::TypeSwitch<Operation *, LogicalResult>(op)
              .Case<linalg::MatmulOp>(
                  [&](linalg::MatmulOp matmul) { return lowerMatmul(matmul); })
              .Default([&](Operation *) { return lowerElementwise(op); });
      if (failed(result)) {
        signalPassFailure();
        return;
      }
    }
  }

  /// Generic lowering for any elementwise linalg op (all-parallel iterators,
  /// identity indexing maps): wraps nested loops in a gpu.launch and clones
  /// the op's scalar body, mapping block args to memref loads and the yield
  /// to a memref store. Covers linalg.generic (relu), linalg.add, etc.
  LogicalResult lowerElementwise(linalg::LinalgOp linalgOp) {
    Operation *op = linalgOp;
    if (op->getNumResults() != 0)
      return op->emitOpError()
             << "expected bufferized memref form before GPU lowering";

    // Must be elementwise: all iterators parallel, all maps identity.
    for (utils::IteratorType it : linalgOp.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return op->emitOpError()
               << "unsupported cuda linalg op: non-parallel iterator";
    for (AffineMap map : linalgOp.getIndexingMapsArray())
      if (!map.isIdentity())
        return op->emitOpError()
               << "unsupported cuda linalg op: non-identity indexing map";

    Value out = linalgOp.getDpsInitOperand(0)->get();
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!outType || !outType.hasStaticShape())
      return op->emitOpError() << "expected static-shape memref output";

    OpBuilder builder(op);
    Location loc = op->getLoc();

    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);

    gpu::LaunchOp launch =
        gpu::LaunchOp::create(builder, loc, c1, c1, c1, c1, c1, c1);
    launch->setAttr("device", builder.getStringAttr("cuda"));

    Block &launchBody = launch.getBody().front();
    builder.setInsertionPointToEnd(&launchBody);
    gpu::TerminatorOp::create(builder, loc);
    builder.setInsertionPointToStart(&launchBody);

    // One scf.for per output dimension.
    SmallVector<Value> ivs;
    for (int64_t dim : outType.getShape()) {
      Value ub = arith::ConstantIndexOp::create(builder, loc, dim);
      scf::ForOp loop = scf::ForOp::create(builder, loc, c0, ub, c1);
      builder.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
    }

    // Map scalar block args to loads: inputs first, then the init/output.
    Block &body = linalgOp->getRegion(0).front();
    IRMapping mapping;
    unsigned argIdx = 0;
    for (OpOperand *input : linalgOp.getDpsInputOperands())
      mapping.map(body.getArgument(argIdx++),
                  memref::LoadOp::create(builder, loc, input->get(), ivs));
    mapping.map(body.getArgument(argIdx),
                memref::LoadOp::create(builder, loc, out, ivs));

    // Clone the scalar body; yield becomes a store to the output.
    for (Operation &bodyOp : body.without_terminator())
      builder.clone(bodyOp, mapping);
    auto yield = cast<linalg::YieldOp>(body.getTerminator());
    Value result = mapping.lookupOrDefault(yield.getOperand(0));
    memref::StoreOp::create(builder, loc, result, out, ivs);

    op->erase();
    return success();
  }

  LogicalResult lowerMatmul(linalg::MatmulOp matmul) {
    if (matmul->getNumResults() != 0)
      return matmul.emitOpError()
             << "expected bufferized memref form before GPU lowering";

    Value lhs = matmul.getInputs()[0];
    Value rhs = matmul.getInputs()[1];
    Value out = matmul.getOutputs()[0];

    auto lhsType = dyn_cast<MemRefType>(lhs.getType());
    auto rhsType = dyn_cast<MemRefType>(rhs.getType());
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!lhsType || !rhsType || !outType)
      return matmul.emitOpError()
             << "expected memref operands before GPU lowering";

    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
        !outType.hasStaticShape())
      return matmul.emitOpError()
             << "only static-shape matmul is supported for now";

    if (outType.getRank() != 2 || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2)
      return matmul.emitOpError() << "expected rank-2 matmul operands";

    int64_t m = outType.getDimSize(0);
    int64_t n = outType.getDimSize(1);
    int64_t k = lhsType.getDimSize(1);
    if (rhsType.getDimSize(0) != k)
      return matmul.emitOpError() << "matmul K dimension mismatch";

    OpBuilder builder(matmul);
    Location loc = matmul.getLoc();

    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
    Value cM = arith::ConstantIndexOp::create(builder, loc, m);
    Value cN = arith::ConstantIndexOp::create(builder, loc, n);
    Value cK = arith::ConstantIndexOp::create(builder, loc, k);

    gpu::LaunchOp launch =
        gpu::LaunchOp::create(builder, loc, c1, c1, c1, c1, c1, c1);
    launch->setAttr("device", builder.getStringAttr("cuda"));

    Block &body = launch.getBody().front();
    builder.setInsertionPointToEnd(&body);
    gpu::TerminatorOp::create(builder, loc);
    builder.setInsertionPointToStart(&body);

    scf::ForOp iLoop = scf::ForOp::create(builder, loc, c0, cM, c1);
    builder.setInsertionPointToStart(iLoop.getBody());
    Value i = iLoop.getInductionVar();

    scf::ForOp jLoop = scf::ForOp::create(builder, loc, c0, cN, c1);
    builder.setInsertionPointToStart(jLoop.getBody());
    Value j = jLoop.getInductionVar();

    scf::ForOp kLoop = scf::ForOp::create(builder, loc, c0, cK, c1);
    builder.setInsertionPointToStart(kLoop.getBody());
    Value kk = kLoop.getInductionVar();

    Value a = memref::LoadOp::create(builder, loc, lhs, ValueRange{i, kk});
    Value b = memref::LoadOp::create(builder, loc, rhs, ValueRange{kk, j});
    Value current = memref::LoadOp::create(builder, loc, out, ValueRange{i, j});
    Value prod = arith::MulFOp::create(builder, loc, a, b);
    Value sum = arith::AddFOp::create(builder, loc, current, prod);
    memref::StoreOp::create(builder, loc, sum, out, ValueRange{i, j});

    matmul.erase();
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hexir::createCudaGpuLoweringPass() {
  return std::make_unique<CudaGpuLoweringPass>();
}
