get_property(hexir_libs GLOBAL PROPERTY HEXIR_LIBS)

add_executable(hexir hexir.cpp)
llvm_update_compile_flags(hexir)
target_include_directories(hexir PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include)

# Static libraries with mutual references: let the linker iterate.
target_link_libraries(hexir PRIVATE
  -Wl,--start-group ${hexir_libs} -Wl,--end-group
  ${HEXIR_MLIR_LIBS})

# Keep the binary at <build>/hexir: test/lit.cfg.py, the readme and docs all
# refer to ./build/hexir.
set_target_properties(hexir PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

install(TARGETS hexir RUNTIME DESTINATION bin)
