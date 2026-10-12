cmake_minimum_required(VERSION 3.20)
project(ninja LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Which LLVM to build against, in order of preference:
#
#   1. -DLLVM_DIR=... -DMLIR_DIR=... on the command line
#   2. the pinned submodule, once scripts/build-llvm.sh has built it
#   3. a machine-wide checkout at /home/user/llvm-project
set(NINJA_LLVM_SUBMODULE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/build")
if(NOT DEFINED LLVM_DIR)
  if(EXISTS "${NINJA_LLVM_SUBMODULE}/lib/cmake/llvm/LLVMConfig.cmake")
    set(LLVM_DIR "${NINJA_LLVM_SUBMODULE}/lib/cmake/llvm")
  else()
    set(LLVM_DIR "/home/user/llvm-project/build/lib/cmake/llvm")
  endif()
endif()
if(NOT DEFINED MLIR_DIR)
  if(EXISTS "${NINJA_LLVM_SUBMODULE}/lib/cmake/mlir/MLIRConfig.cmake")
    set(MLIR_DIR "${NINJA_LLVM_SUBMODULE}/lib/cmake/mlir")
  else()
    set(MLIR_DIR "/home/user/llvm-project/build/lib/cmake/mlir")
  endif()
endif()

find_package(MLIR REQUIRED CONFIG)
message(STATUS "ninja: MLIR from ${MLIR_DIR} (LLVM ${LLVM_PACKAGE_VERSION})")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)   # mlir_tablegen(), add_public_tablegen_target()
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

# include/ runs TableGen (.td -> .inc), lib/ compiles the dialect, tools/ links
# ninja-opt.
add_subdirectory(include/ninja)
add_subdirectory(lib)
add_subdirectory(tools/ninja-opt)

find_program(LLVM_LIT_EXECUTABLE NAMES llvm-lit lit)
if(LLVM_LIT_EXECUTABLE)
  add_custom_target(check-ninja
    COMMAND ${LLVM_LIT_EXECUTABLE} -v --param build_dir=${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/test
    DEPENDS ninja-opt
    COMMENT "Running ninja tests"
    USES_TERMINAL)
else()
  message(WARNING "lit not found: check-ninja unavailable (pip install lit)")
endif()
