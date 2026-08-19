# The hexir runtime.
#
# Deliberately independent of the compiler: plain C99, no MLIR, no LLVM, and
# no include path into ../include. That is the whole point of shipping a
# module -- deploying must not mean shipping the compiler. Keep it that way;
# if this ever needs to link an MLIR library, the layering is wrong.
#
# Builds standalone too:  cmake -S runtime -B build-runtime && cmake --build build-runtime
cmake_minimum_required(VERSION 3.13.4)
project(hexir_runtime LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED YES)

add_library(hexir_runtime STATIC
  src/status.c
  src/hal/hal.c
  src/hal/cpu/cpu_device.c
  src/module/module.c
  src/vm/interpreter.c
  src/kernels/reference_kernels.c)

target_include_directories(hexir_runtime
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_options(hexir_runtime PRIVATE -Wall -Wextra)

add_executable(hexir-run tools/hexir-run/hexir-run.c)
target_link_libraries(hexir-run PRIVATE hexir_runtime)
target_compile_options(hexir-run PRIVATE -Wall -Wextra)

if(DEFINED HEXIR_TOOLS_OUTPUT_DIRECTORY)
  set_target_properties(hexir-run PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${HEXIR_TOOLS_OUTPUT_DIRECTORY})
endif()

install(TARGETS hexir-run RUNTIME DESTINATION bin)
install(TARGETS hexir_runtime ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
