#include "hexir/Dialect/Hexir/IR/HexirDialect.h"
#include "hexir/Target/TargetInfo.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace mlir {
namespace hexir {

struct PartitionPass
    : public PassWrapper<PartitionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionPass)

  StringRef getArgument() const final { return "hexir-partition"; }
  StringRef getDescription() const final {
    return "Partition Hexir and Linalg ops to CPU/GPU based on target support.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    TargetSupport &targets = TargetSupport::getInstance();

    auto assignDevice = [&targets](Operation *op) -> StringRef {
      StringRef preferred = targets.getPreferredTarget(op);
      if (targets.isSupported(op, preferred))
        return preferred;
      return "cpu";
    };

    
    module->setAttr("hexir.targets",
                    ArrayAttr::get(ctx, {StringAttr::get(ctx, "cpu"),
                                         StringAttr::get(ctx, "cuda")}));

    module->walk([ctx, &assignDevice](Operation *op) {
      // Respect placement decided earlier in the pipeline: the pass runs
      // once BEFORE LowerToLinalg (annotating hexir ops, e.g. hexir.linear)
      // and once after as a fallback. Propagated attrs win.
      if (op->hasAttr("device"))
        return;

      // Hexir dialect ops (first run: hexir.linear / hexir.relu / ...).
      if (isa<HexirDialect>(op->getDialect())) {
        StringRef device = assignDevice(op);
        if (!device.empty()) {
          op->setAttr("device", StringAttr::get(ctx, device));
        }
        return;
      }

      // ALL linalg ops, generically (second run): fallback for ops that did
      // not inherit a device attr from a hexir op during lowering.
      if (isa<linalg::LinalgDialect>(op->getDialect())) {
        op->setAttr("device", StringAttr::get(ctx, assignDevice(op)));
        return;
      }
    });
  }
};

std::unique_ptr<Pass> createPartitionPass() {
  return std::make_unique<PartitionPass>();
}

} // namespace hexir
} // namespace mlir
