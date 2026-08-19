set(LLVM_TARGET_DEFINITIONS HexirOps.td)
mlir_tablegen(HexirOps.h.inc -gen-op-decls)
mlir_tablegen(HexirOps.cpp.inc -gen-op-defs)
mlir_tablegen(HexirDialect.h.inc -gen-dialect-decls)
mlir_tablegen(HexirDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(HexirOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(HexirShapeInferenceIncGen)

# DRR rewrites used by canonicalization.
set(LLVM_TARGET_DEFINITIONS HexirCombine.td)
mlir_tablegen(HexirCombine.inc -gen-rewriters)
add_public_tablegen_target(HexirCombineIncGen)
