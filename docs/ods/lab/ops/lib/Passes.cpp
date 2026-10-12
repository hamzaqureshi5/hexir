#include "ninja/NinjaDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::ninja {
#define GEN_PASS_DECL
#include "ninja/Passes.h.inc"

#define GEN_PASS_DEF_NINJACOUNTSTRIKES
#include "ninja/Passes.h.inc"

namespace {
struct CountStrikes : impl::NinjaCountStrikesBase<CountStrikes> {
  using NinjaCountStrikesBase::NinjaCountStrikesBase;
  void runOnOperation() override {
    int n = 0;
    getOperation().walk([&](StrikeOp) { ++n; });
    if (verbose)
      llvm::errs() << getOperation().getSymName() << ": ";
    llvm::errs() << n << " strikes\n";
  }
};
} // namespace

#define GEN_PASS_REGISTRATION
#include "ninja/Passes.h.inc"
} // namespace mlir::ninja
void registerNinjaPasses() { mlir::ninja::registerNinjaPasses(); }
