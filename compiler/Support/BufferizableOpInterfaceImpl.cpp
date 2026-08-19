//===- BufferizableOpInterfaceImpl.cpp - Hexir bufferization ---*- C++ -*-===//
//
// One-Shot Bufferize needs to know how to convert every op that touches
// tensors. `hexir.print` reads a tensor and produces no result, so without an
// interface implementation bufferization either fails outright ("op was not
// bufferized") or -- with allowUnknownOps -- leaves the op on a tensor and
// materializes a `bufferization.to_tensor` to feed it, which the printf
// lowering in LowerToLLVM.cpp cannot consume (it casts the operand to
// MemRefType).
//
//===----------------------------------------------------------------------===//

#include "hexir/Support/BufferizableOpInterfaceImpl.h"
#include "hexir/Dialect/Hexir/IR/HexirDialect.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {

/// Bufferization of `hexir.print`: read-only, no results. Replaced with a new
/// `hexir.print` operating on the buffer.
struct PrintOpInterface
    : public BufferizableOpInterface::ExternalModel<PrintOpInterface,
                                                    hexir::PrintOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // print only ever reads its operand.
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    // No results, so nothing aliases the operand.
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto printOp = cast<hexir::PrintOp>(op);

    // Already lowered to a memref by an earlier partial conversion.
    if (!isa<TensorType>(printOp.getInput().getType()))
      return success();

    FailureOr<Value> buffer =
        getBuffer(rewriter, printOp.getInput(), options, state);
    if (failed(buffer))
      return failure();

    replaceOpWithNewBufferizedOp<hexir::PrintOp>(rewriter, printOp, *buffer);
    return success();
  }
};

} // namespace

void mlir::hexir::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, hexir::HexirDialect *dialect) {
    hexir::PrintOp::attachInterface<PrintOpInterface>(*ctx);
  });
}
