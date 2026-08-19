//====- LowerToAffineLoops.cpp - Partial lowering from Toy to Affine+Std --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a combination of
// affine loops, memref operations and standard operations. This lowering
// expects that all calls have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "hexir/Dialect/Hexir/IR/HexirDialect.h"
#include "hexir/Conversion/Passes.h"
#include "hexir/Dialect/Hexir/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert the given RankedTensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(RankedTensorType type)
{
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter)
{
  auto alloc = memref::AllocOp::create(rewriter, loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as hexir functions have no control flow.
  auto dealloc = memref::DeallocOp::create(rewriter, loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder and the range of loop induction
/// variables for the iteration. It returns a value to store at the current
/// index of the iteration.
using LoopIterationFn =
    function_ref<Value(OpBuilder &rewriter, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, PatternRewriter &rewriter,
                           LoopIterationFn processIteration)
{
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs)
      {
        // Call the processing function with the rewriter
        // and the loop induction variables. This function will return the value
        // to store at the current index.
        Value valueToStore = processIteration(nestedBuilder, ivs);
        affine::AffineStoreOp::create(nestedBuilder, loc, valueToStore, alloc,
                                      ivs);
      });

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

namespace
{
  //===----------------------------------------------------------------------===//
  // HexirToAffine Conversion Patterns: Binary operations
  //===----------------------------------------------------------------------===//

  template <typename BinaryOp, typename LoweredBinaryOp>
  struct BinaryOpLowering : public OpConversionPattern<BinaryOp>
  {
    using OpConversionPattern<BinaryOp>::OpConversionPattern;
    using OpAdaptor = typename OpConversionPattern<BinaryOp>::OpAdaptor;

    LogicalResult
    matchAndRewrite(BinaryOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const final
    {
      auto loc = op->getLoc();
      lowerOpToLoops(op, rewriter, [&](OpBuilder &builder, ValueRange loopIvs)
                     {
      // Generate loads for the element of 'lhs' and 'rhs' at the
      // inner loop.
      auto loadedLhs =
          affine::AffineLoadOp::create(builder, loc, adaptor.getLhs(), loopIvs);
      auto loadedRhs =
          affine::AffineLoadOp::create(builder, loc, adaptor.getRhs(), loopIvs);

      // Create the binary operation performed on the loaded
      // values.
      return LoweredBinaryOp::create(builder, loc, loadedLhs, loadedRhs); });
      return success();
    }
  };

  using AddOpLowering = BinaryOpLowering<hexir::AddOp, arith::AddFOp>;
  // using MulOpLowering = BinaryOpLowering<hexir::MulOp, arith::MulFOp>;

  //===----------------------------------------------------------------------===//
  // HexirToAffine Conversion Patterns: Constant operations
  //===----------------------------------------------------------------------===//

  struct ConstantOpLowering : public OpConversionPattern<hexir::ConstantOp>
  {
    using OpConversionPattern<hexir::ConstantOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(hexir::ConstantOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const final
    {
      DenseElementsAttr constantValue = op.getValue();
      Location loc = op.getLoc();

      // When lowering the constant operation, we allocate and assign the constant
      // values to a corresponding memref allocation.
      auto tensorType = llvm::cast<RankedTensorType>(op.getType());
      auto memRefType = convertTensorToMemRef(tensorType);
      auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

      // We will be generating constant indices up-to the largest dimension.
      // Create these constants up-front to avoid large amounts of redundant
      // operations.
      auto valueShape = memRefType.getShape();
      SmallVector<Value, 8> constantIndices;

      if (!valueShape.empty())
      {
        for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape)))
          constantIndices.push_back(
              arith::ConstantIndexOp::create(rewriter, loc, i));
      }
      else
      {
        // This is the case of a tensor of rank 0.
        constantIndices.push_back(
            arith::ConstantIndexOp::create(rewriter, loc, 0));
      }

      // The constant operation represents a multi-dimensional constant, so we
      // will need to generate a store for each of the elements. The following
      // functor recursively walks the dimensions of the constant shape,
      // generating a store when the recursion hits the base case.
      SmallVector<Value, 2> indices;
      auto valueIt = constantValue.value_begin<FloatAttr>();
      std::function<void(uint64_t)> storeElements = [&](uint64_t dimension)
      {
        // The last dimension is the base case of the recursion, at this point
        // we store the element at the given index.
        if (dimension == valueShape.size())
        {
          affine::AffineStoreOp::create(
              rewriter, loc, arith::ConstantOp::create(rewriter, loc, *valueIt++),
              alloc, llvm::ArrayRef(indices));
          return;
        }

        // Otherwise, iterate over the current dimension and add the indices to
        // the list.
        for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i)
        {
          indices.push_back(constantIndices[i]);
          storeElements(dimension + 1);
          indices.pop_back();
        }
      };

      // Start the element storing recursion from the first dimension.
      storeElements(/*dimension=*/0);

      // Replace this operation with the generated alloc.
      rewriter.replaceOp(op, alloc);
      return success();
    }
  };

  //===----------------------------------------------------------------------===//
  // HexirToAffine Conversion Patterns: Func operations
  //===----------------------------------------------------------------------===//

  struct FuncOpLowering : public OpConversionPattern<hexir::FuncOp>
  {
    using OpConversionPattern<hexir::FuncOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(hexir::FuncOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const final
    {
      // We only lower the main function as we expect that all other functions
      // have been inlined.
      if (op.getName() != "main")
        return failure();

      // Verify that the given main has no inputs and results.
      if (op.getNumArguments() || op.getFunctionType().getNumResults())
      {
        return rewriter.notifyMatchFailure(op, [](Diagnostic &diag)
                                           { diag << "expected 'main' to have 0 inputs and 0 results"; });
      }

      // Create a new non-hexir function, with the same region.
      auto func = mlir::func::FuncOp::create(rewriter, op.getLoc(), op.getName(),
                                             op.getFunctionType());
      rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
      rewriter.eraseOp(op);
      return success();
    }
  };

  //===----------------------------------------------------------------------===//
  // HexirToAffine Conversion Patterns: Print operations
  //===----------------------------------------------------------------------===//

  struct PrintOpLowering : public OpConversionPattern<hexir::PrintOp>
  {
    using OpConversionPattern<hexir::PrintOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(hexir::PrintOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const final
    {
      // We don't lower "hexir.print" in this pass, but we need to update its
      // operands.
      rewriter.modifyOpInPlace(op,
                               [&]
                               { op->setOperands(adaptor.getOperands()); });
      return success();
    }
  };

  // ===----------------------------------------------------------------------===//
  // HexirToAffine Conversion Patterns: Return operations
  // ===----------------------------------------------------------------------===//

  // struct ReturnOpLowering : public OpConversionPattern<hexir::ReturnOp> {
  //   using OpConversionPattern<hexir::ReturnOp>::OpConversionPattern;

  //   LogicalResult
  //   matchAndRewrite(hexir::ReturnOp op, OpAdaptor adaptor,
  //                   ConversionPatternRewriter &rewriter) const final {
  //     // During this lowering, we expect that all function calls have been
  //     // inlined.
  //     if (op.hasOperand())
  //       return failure();

  //     // We lower "hexir.return" directly to "func.return".
  //     rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
  //     return success();
  //   }
  // };

  // ===----------------------------------------------------------------------===//
  // HexirToAffine Conversion Patterns: Transpose operations
  // ===----------------------------------------------------------------------===//

  // struct TransposeOpLowering : public OpConversionPattern<hexir::TransposeOp> {
  //   using OpConversionPattern<hexir::TransposeOp>::OpConversionPattern;

  //   LogicalResult
  //   matchAndRewrite(hexir::TransposeOp op, OpAdaptor adaptor,
  //                   ConversionPatternRewriter &rewriter) const final {
  //     auto loc = op->getLoc();
  //     lowerOpToLoops(op, rewriter, [&](OpBuilder &builder, ValueRange loopIvs)
  //     {
  //       Value input = adaptor.getInput();

  //       // Transpose the elements by generating a load from the
  //       // reverse indices.
  //       SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
  //       return affine::AffineLoadOp::create(builder, loc, input, reverseIvs);
  //     });
  //     return success();
  //   }
  // };

} // namespace

//===----------------------------------------------------------------------===//
// HexirToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the hexir operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Hexir dialect.
namespace
{
  struct HexirToAffineLoweringPass
      : public PassWrapper<HexirToAffineLoweringPass, OperationPass<ModuleOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HexirToAffineLoweringPass)
    StringRef getArgument() const override { return "hexir-to-affine"; }

    void getDependentDialects(DialectRegistry &registry) const override
    {
      registry.insert<affine::AffineDialect, func::FuncDialect,
                      memref::MemRefDialect>();
    }
    void runOnOperation() final;
  };
} // namespace

void HexirToAffineLoweringPass::runOnOperation()
{
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect>();

  // We also define the Hexir dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Hexir operations that don't want
  // to lower, `hexir.print`, as `legal`. `hexir.print` will still need its operands
  // to be updated though (as we convert from TensorType to MemRefType), so we
  // only treat it as `legal` if its operands are legal.
  target.addIllegalDialect<hexir::HexirDialect>();
  target.addDynamicallyLegalOp<hexir::PrintOp>([](hexir::PrintOp op)
                                             { return llvm::none_of(op->getOperandTypes(),
                                                                    [](Type type)
                                                                    { return llvm::isa<TensorType>(type); }); });

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Hexir operations.
  RewritePatternSet patterns(&getContext());
  patterns
      .add<
          AddOpLowering, ConstantOpLowering, FuncOpLowering, PrintOpLowering /*,MulOpLowering, ReturnOpLowering, TransposeOpLowering*/>(
          &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

// Create a pass for lowering operations in the `Affine` and `Std` dialects, for
// a subset of the Hexir IR (e.g. matmul).
std::unique_ptr<Pass> mlir::hexir::createLowerToAffinePass()
{
  return std::make_unique<HexirToAffineLoweringPass>();
}
