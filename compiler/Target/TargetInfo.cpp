#include "hexir/Target/TargetInfo.h"
#include "hexir/Dialect/Hexir/IR/HexirDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace hexir {

TargetSupport &TargetSupport::getInstance() {
  static TargetSupport instance;
  return instance;
}

// "gpu" is accepted everywhere as an alias for "cuda".
static StringRef normalizeTarget(StringRef target) {
  return target == "gpu" ? StringRef("cuda") : target;
}

TargetSupport::TargetSupport() {
  registerSupport("hexir.linear", {"cpu", "cuda"});
  registerSupport("hexir.add", {"cpu", "cuda"});
  registerSupport("hexir.relu", {"cpu", "cuda"});

  registerSupport("linalg.matmul", {"cpu", "cuda"});
  registerSupport("linalg.add", {"cpu", "cuda"});
  registerSupport("linalg.generic", {"cpu", "cuda"});

  // ── Primary placement knobs ────────────────────────────────────────────
  // Placement is decided on the frontend hexir ops (PartitionPass runs
  // before LowerToLinalg) and the device attr is propagated down through
  // every later lowering. Flip these — or use the -placement flag — to
  // reroute any op:
  //   hexir.linear = "cpu" | "cuda"   (the matmul)
  //   hexir.relu   = "cpu" | "cuda"
  opPreferred_["hexir.linear"] = "cpu"; // compute-intensive: GPU by default
  opPreferred_["hexir.add"] = "cpu";
  opPreferred_["hexir.relu"] = "cpu";

  // Fallbacks for linalg ops that did not originate from an annotated hexir
  // op (a second PartitionPass run fills in anything still unannotated).
  opPreferred_["linalg.matmul"] = "cpu";
  opPreferred_["linalg.add"] = "cpu";
  opPreferred_["linalg.generic"] = "cpu";
}

bool TargetSupport::isSupported(Operation *op, StringRef target) const {
  StringRef t = normalizeTarget(target);
  auto it = opSupports_.find(op->getName().getStringRef());
  if (it == opSupports_.end())
    return t == "cpu";
  return it->second.contains(t);
}

StringRef TargetSupport::getPreferredTarget(Operation *op) const {
  auto it = opPreferred_.find(op->getName().getStringRef());
  if (it == opPreferred_.end())
    return "cpu";
  return it->second;
}

void TargetSupport::registerSupport(StringRef opName,
                                    llvm::ArrayRef<StringRef> targets) {
  llvm::StringSet<> &opSet = opSupports_[opName];
  for (StringRef target : targets)
    opSet.insert(normalizeTarget(target));
}

bool TargetSupport::isKnownOp(StringRef opName) const {
  return opSupports_.find(opName) != opSupports_.end();
}

std::vector<std::string> TargetSupport::knownOps() const {
  std::vector<std::string> names;
  for (const auto &entry : opSupports_)
    names.push_back(entry.first().str());
  llvm::sort(names);
  return names;
}

std::vector<std::string> TargetSupport::targetsFor(StringRef opName) const {
  std::vector<std::string> names;
  auto it = opSupports_.find(opName);
  if (it == opSupports_.end())
    return names;
  for (const auto &target : it->second)
    names.push_back(target.first().str());
  llvm::sort(names);
  return names;
}

bool TargetSupport::setPreferredTarget(StringRef opName, StringRef target) {
  StringRef t = normalizeTarget(target);
  auto it = opSupports_.find(opName);
  if (it == opSupports_.end() || !it->second.contains(t))
    return false;
  opPreferred_[opName] = t.str();
  return true;
}

} // namespace hexir
} // namespace mlir
