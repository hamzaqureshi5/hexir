//===- ModuleSerializer.h - Emit a loadable module -------------*- C++ -*-===//
//
// Writes the .hxb container the runtime loads. Input is the IR at the kernel
// level (Stage::TIR): hexir.call_tir against hextir.prim_func.
//
//===----------------------------------------------------------------------===//

#ifndef HEXIR_SERIALIZATION_MODULESERIALIZER_H
#define HEXIR_SERIALIZATION_MODULESERIALIZER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace hexir {

/// Serialize `module` to the runtime's container format. `module` must be at
/// the kernel level -- run the pipeline to Stage::TIR first.
LogicalResult serializeToHXB(ModuleOp module, llvm::raw_ostream &os);

} // namespace hexir
} // namespace mlir

#endif // HEXIR_SERIALIZATION_MODULESERIALIZER_H
