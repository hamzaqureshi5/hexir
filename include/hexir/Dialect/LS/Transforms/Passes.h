//===- Passes.h - LS dialect transforms ------------------------*- C++ -*-===//

#ifndef HEXIR_DIALECT_LS_TRANSFORMS_PASSES_H
#define HEXIR_DIALECT_LS_TRANSFORMS_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace hexir {

/// Rewrite linalg ops into ls_cpu.*/ls_gpu.* mirror ops so that placement is
/// visible in the IR. Purely presentational -- LSToLinalg converts them back.
std::unique_ptr<Pass> createMaterializeLSTargetsPass();

} // namespace hexir
} // namespace mlir

#endif // HEXIR_DIALECT_LS_TRANSFORMS_PASSES_H
