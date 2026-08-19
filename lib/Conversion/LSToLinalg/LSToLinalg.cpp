//===- LowerLSToLinalg.cpp - Lower ls_cpu/ls_gpu ops to Linalg ------------===//
//
// Lowers the lightweight ls_cpu and ls_gpu model dialect ops back to standard
// linalg operations so that the existing downstream passes (bufferization,
// CudaGpuLowering, linalg-to-loops) can proceed unchanged.
//
// Device attributes are preserved on the replacement ops so that
// CudaGpuLoweringPass still picks up ls_gpu.matmul → linalg.matmul{device=cuda}.
//===----------------------------------------------------------------------===//

#include "hexir/Dialect/Hexir/IR/HexirDialect.h"
#include "hexir/Dialect/LS/IR/LSDialects.h"
#include "hexir/Conversion/Passes.h"
#include "hexir/Dialect/Hexir/Transforms/Passes.h"
#include "hexir/Dialect/LS/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <memory>

using namespace mlir;

namespace {

// Helper: build a linalg.matmul on tensor operands, optionally tagging it with
// a device attribute (so CudaGpuLoweringPass can find it later).
static Value buildMatmul(ConversionPatternRewriter &rewriter, Location loc,
                         Value lhs, Value rhs, RankedTensorType resultTy,
                         StringRef device) {
  // Zero-initialized destination for the accumulation. tensor.empty +
  // linalg.fill, not arith.constant: a constant is read-only, so using it as a
  // destination-passing `outs` operand makes bufferization emit a read-only
  // global plus alloc_tensor(copy) to make it writable, which then fails to
  // bufferize. tensor.empty becomes a plain memref.alloc.
  Value zero = arith::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(resultTy.getElementType()));
  Value empty = tensor::EmptyOp::create(rewriter, loc, resultTy.getShape(),
                                        resultTy.getElementType());
  Value init =
      linalg::FillOp::create(rewriter, loc, ValueRange{zero}, ValueRange{empty})
          .getResult(0);

  auto matmul = linalg::MatmulOp::create(rewriter, loc,
                                         TypeRange{resultTy},
                                         ValueRange{lhs, rhs},
                                         ValueRange{init});
  matmul->setAttr("device", StringAttr::get(rewriter.getContext(), device));
  return matmul.getResult(0);
}

// Helper: build a linalg.generic implementing element-wise ReLU (maximumf 0).
static Value buildRelu(ConversionPatternRewriter &rewriter, Location loc,
                       Value input, RankedTensorType resultTy,
                       StringRef device) {
  // Destination for the element-wise result. linalg.generic overwrites every
  // element, so an uninitialized tensor.empty is sufficient here (see
  // buildMatmul for why this must not be an arith.constant).
  Value init = tensor::EmptyOp::create(rewriter, loc, resultTy.getShape(),
                                       resultTy.getElementType());

  auto identity = rewriter.getMultiDimIdentityMap(resultTy.getRank());
  SmallVector<AffineMap, 2> indexingMaps = {identity, identity};
  SmallVector<utils::IteratorType> iteratorTypes(resultTy.getRank(),
                                                 utils::IteratorType::parallel);

  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, TypeRange{resultTy}, ValueRange{input}, ValueRange{init},
      indexingMaps, iteratorTypes,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        Value x = args[0];
        Value zero = arith::ConstantOp::create(
            builder, loc, builder.getFloatAttr(x.getType(), 0.0));
        Value relu = arith::MaximumFOp::create(builder, loc, x, zero);
        linalg::YieldOp::create(builder, loc, relu);
      });
  genericOp->setAttr("device", StringAttr::get(rewriter.getContext(), device));
  return genericOp.getResult(0);
}

//===----------------------------------------------------------------------===//
// ls_gpu.matmul → linalg.matmul {device = "cuda"}
//===----------------------------------------------------------------------===//
struct LSGpuMatmulToLinalg : public OpConversionPattern<ls_gpu::MatmulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ls_gpu::MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultTy = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

    rewriter.replaceOp(op, buildMatmul(rewriter, op.getLoc(),
                                       adaptor.getLhs(), adaptor.getRhs(),
                                       resultTy, "cuda"));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ls_cpu.matmul → linalg.matmul {device = "cpu"}
//===----------------------------------------------------------------------===//
struct LSCpuMatmulToLinalg : public OpConversionPattern<ls_cpu::MatmulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ls_cpu::MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultTy = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

    rewriter.replaceOp(op, buildMatmul(rewriter, op.getLoc(),
                                       adaptor.getLhs(), adaptor.getRhs(),
                                       resultTy, "cpu"));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ls_cpu.relu → linalg.generic {device = "cpu"}
//===----------------------------------------------------------------------===//
struct LSCpuReluToLinalg : public OpConversionPattern<ls_cpu::ReluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ls_cpu::ReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultTy = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

    rewriter.replaceOp(op, buildRelu(rewriter, op.getLoc(),
                                     adaptor.getInput(), resultTy, "cpu"));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ls_gpu.relu → linalg.generic {device = "cuda"}
//===----------------------------------------------------------------------===//
struct LSGpuReluToLinalg : public OpConversionPattern<ls_gpu::ReluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ls_gpu::ReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultTy = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

    rewriter.replaceOp(op, buildRelu(rewriter, op.getLoc(),
                                     adaptor.getInput(), resultTy, "cuda"));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//
struct LSTargetsToLinalgLoweringPass
    : public PassWrapper<LSTargetsToLinalgLoweringPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LSTargetsToLinalgLoweringPass)

  StringRef getArgument() const override { return "ls-lower-to-linalg"; }
  StringRef getDescription() const override {
    return "Lower ls_cpu/ls_gpu model ops to the Linalg dialect.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect,
                           tensor::TensorDialect, func::FuncDialect,
                           memref::MemRefDialect, BuiltinDialect>();
    target.addIllegalDialect<ls_cpu::LSCPUDialect, ls_gpu::LSGPUDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<LSGpuMatmulToLinalg, LSCpuMatmulToLinalg,
                 LSCpuReluToLinalg,   LSGpuReluToLinalg>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace hexir {

std::unique_ptr<mlir::Pass> createLSTargetsToLinalgPass() {
  return std::make_unique<LSTargetsToLinalgLoweringPass>();
}

} // namespace hexir
} // namespace mlir
