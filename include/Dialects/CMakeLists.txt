# Most dialects should use add_mlir_dialect().  See examples/standalone.
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(OpsIncGen)

set(LLVM_TARGET_DEFINITIONS LSDialects.td)
mlir_tablegen(LSDialectsOps.h.inc -gen-op-decls)
mlir_tablegen(LSDialectsOps.cpp.inc -gen-op-defs)
mlir_tablegen(LSCPUDialect.h.inc -gen-dialect-decls -dialect=ls_cpu)
mlir_tablegen(LSCPUDialect.cpp.inc -gen-dialect-defs -dialect=ls_cpu)

mlir_tablegen(LSGPUDialect.h.inc -gen-dialect-decls -dialect=ls_gpu)
mlir_tablegen(LSGPUDialect.cpp.inc -gen-dialect-defs -dialect=ls_gpu)
add_public_tablegen_target(LSDialectsIncGen)

set(LLVM_TARGET_DEFINITIONS HexTIROps.td)
mlir_tablegen(HexTIROps.h.inc -gen-op-decls)
mlir_tablegen(HexTIROps.cpp.inc -gen-op-defs)
mlir_tablegen(HexTIRDialect.h.inc -gen-dialect-decls -dialect=hextir)
mlir_tablegen(HexTIRDialect.cpp.inc -gen-dialect-defs -dialect=hextir)
add_public_tablegen_target(HexTIRIncGen)

# Most dialects should use add_mlir_interfaces().
set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ShapeInferenceInterfaceIncGen)
