//===- MaterializeLSTargets.cpp - Build ls_cpu/ls_gpu IR -------*- C++ -*-===//

#include "hexir/Dialect/LS/IR/LSDialects.h"
#include "hexir/Conversion/Passes.h"
#include "hexir/Dialect/Hexir/Transforms/Passes.h"
#include "hexir/Dialect/LS/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

using namespace mlir;

namespace {

static bool isGpu(Operation *op) {
  auto device = op->getAttrOfType<StringAttr>("device");
  return device && (device.getValue() == "cuda" || device.getValue() == "gpu");
}

template <typename CPUOp, typename GPUOp>
static void replaceBinary(Operation *op, Value lhs, Value rhs,
                          PatternRewriter &rewriter) {
  Type resultType = op->getResult(0).getType();
  if (isGpu(op))
    rewriter.replaceOpWithNewOp<GPUOp>(op, resultType, lhs, rhs);
  else
    rewriter.replaceOpWithNewOp<CPUOp>(op, resultType, lhs, rhs);
}

template <typename CPUOp, typename GPUOp>
static void replaceUnary(Operation *op, Value input, PatternRewriter &rewriter) {
  Type resultType = op->getResult(0).getType();
  if (isGpu(op))
    rewriter.replaceOpWithNewOp<GPUOp>(op, resultType, input);
  else
    rewriter.replaceOpWithNewOp<CPUOp>(op, resultType, input);
}

struct LinalgAddToLS : public OpRewritePattern<linalg::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::AddOp op,
                                PatternRewriter &rewriter) const override {
    replaceBinary<ls_cpu::AddOp, ls_gpu::AddOp>(
        op, op.getInputs()[0], op.getInputs()[1], rewriter);
    return success();
  }
};

struct LinalgMatmulToLS : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    replaceBinary<ls_cpu::MatmulOp, ls_gpu::MatmulOp>(
        op, op.getInputs()[0], op.getInputs()[1], rewriter);
    return success();
  }
};

struct LinalgGenericToLSRelu : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumDpsInputs() != 1 || op->getNumResults() != 1)
      return failure();

    replaceUnary<ls_cpu::ReluOp, ls_gpu::ReluOp>(
        op, op.getDpsInputOperand(0)->get(), rewriter);
    return success();
  }
};

struct MaterializeLSTargetsPass
    : public PassWrapper<MaterializeLSTargetsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeLSTargetsPass)

  StringRef getArgument() const final { return "hexir-materialize-ls-targets"; }

  StringRef getDescription() const final {
    return "Materialize device annotations as ls_cpu/ls_gpu model operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ls_cpu::LSCPUDialect, ls_gpu::LSGPUDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgAddToLS, LinalgMatmulToLS, LinalgGenericToLSRelu>(
        &getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hexir::createMaterializeLSTargetsPass() {
  return std::make_unique<MaterializeLSTargetsPass>();
}
