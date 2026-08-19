# Benchmark harness. Links the runtime only; cuBLAS is dlopened at run time so
# this still builds with no CUDA present.
add_executable(hexir-bench hexir-bench.c)
target_link_libraries(hexir-bench PRIVATE hexir_runtime m ${CMAKE_DL_LIBS})
target_compile_options(hexir-bench PRIVATE -Wall -Wextra)

if(DEFINED HEXIR_TOOLS_OUTPUT_DIRECTORY)
  set_target_properties(hexir-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${HEXIR_TOOLS_OUTPUT_DIRECTORY})
endif()
