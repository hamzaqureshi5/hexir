//===- Passes.h - Hexir dialect transforms ---------------------*- C++ -*-===//
//
// Passes that rewrite hexir IR into hexir IR. Anything that crosses into
// another dialect belongs in hexir/Conversion instead.
//
//===----------------------------------------------------------------------===//

#ifndef HEXIR_DIALECT_HEXIR_TRANSFORMS_PASSES_H
#define HEXIR_DIALECT_HEXIR_TRANSFORMS_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace hexir {

/// Infer result shapes through the ShapeInferenceOpInterface.
std::unique_ptr<Pass> createShapeInferencePass();

/// Annotate ops with the device they run on, from the TargetSupport registry.
/// Skips ops that already carry a `device` attr, so placement propagated by an
/// earlier lowering wins over a fresh decision.
std::unique_ptr<Pass> createPartitionPass();

} // namespace hexir
} // namespace mlir

#endif // HEXIR_DIALECT_HEXIR_TRANSFORMS_PASSES_H
