#pragma once

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include <string>

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

private:
  TargetSupport();

  llvm::StringMap<llvm::StringSet<>> opSupports_;
  llvm::StringMap<std::string> opPreferred_;
};

} // namespace hexir
} // namespace mlir
