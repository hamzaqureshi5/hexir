add_library(NinjaDialect STATIC NinjaDialect.cpp)
llvm_update_compile_flags(NinjaDialect)
target_link_libraries(NinjaDialect PUBLIC
  MLIRIR MLIRSideEffectInterfaces MLIRInferTypeOpInterface)
# The .inc files must exist before NinjaDialect.cpp compiles.
add_dependencies(NinjaDialect NinjaOpsIncGen)
