add_library(NinjaDialect STATIC NinjaDialect.cpp Passes.cpp)
llvm_update_compile_flags(NinjaDialect)
target_link_libraries(NinjaDialect PUBLIC
  MLIRIR MLIRSideEffectInterfaces MLIRInferTypeOpInterface MLIRPass MLIRFuncDialect)
# The .inc files must exist before NinjaDialect.cpp compiles.
add_dependencies(NinjaDialect NinjaOpsIncGen NinjaPatternsIncGen NinjaPassesIncGen NinjaAttrsIncGen)
