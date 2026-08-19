#pragma once

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include <string>
#include <vector>

namespace mlir {
namespace hexir {

/// Target support info for operators.
class TargetSupport {
public:
  static TargetSupport &getInstance();

  /// Check if op is supported on target (e.g., "cpu", "gpu").
  bool isSupported(Operation *op, StringRef target) const;

  /// Get preferred target for op (empty if none).
  StringRef getPreferredTarget(Operation *op) const;

  /// Register support: opName -> targets.
  void registerSupport(StringRef opName, llvm::ArrayRef<StringRef> targets);

  /// Override the preferred target for an op at runtime (e.g. from the
  /// -placement command-line flag). Returns false if the op does not
  /// support the requested target.
  bool setPreferredTarget(StringRef opName, StringRef target);

  /// Is this an op the registry knows about at all?
  ///
  /// Distinct from isSupported: an unknown op and an op that cannot run on the
  /// requested device are different mistakes and deserve different messages.
  bool isKnownOp(StringRef opName) const;

  /// Every placeable op, sorted, for diagnostics. Naming what is valid is more
  /// use to someone who mistyped than saying their input was invalid.
  std::vector<std::string> knownOps() const;

  /// The devices this op can run on, sorted.
  std::vector<std::string> targetsFor(StringRef opName) const;

private:
  TargetSupport();

  llvm::StringMap<llvm::StringSet<>> opSupports_;
  llvm::StringMap<std::string> opPreferred_;
};

} // namespace hexir
} // namespace mlir
