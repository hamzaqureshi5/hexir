add_executable(ninja-opt ninja-opt.cpp)
llvm_update_compile_flags(ninja-opt)
target_link_libraries(ninja-opt PRIVATE
  NinjaDialect MLIROptLib MLIRFuncDialect MLIRTransforms
  MLIRArithDialect MLIRMemRefDialect MLIRSCFDialect)
# Put the binary at build/ninja-opt, where the tests and README expect it.
set_target_properties(ninja-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
