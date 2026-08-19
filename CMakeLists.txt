cmake_minimum_required(VERSION 3.13.4)
project(hexir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Point these at your LLVM/MLIR build, or pass -DLLVM_DIR=... -DMLIR_DIR=...
if(NOT DEFINED LLVM_DIR)
  set(LLVM_DIR "/home/user/llvm-project/build/lib/cmake/llvm")
endif()
if(NOT DEFINED MLIR_DIR)
  set(MLIR_DIR "/home/user/llvm-project/build/lib/cmake/mlir")
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(AddLLVM)
include(AddMLIR)
include(TableGen)
include(HandleLLVMOptions)
include(HexirLibrary)

set(LLVM_LINK_COMPONENTS Core Support nativecodegen OrcJIT)

add_definitions(${LLVM_DEFINITIONS})

# The runtime is added BEFORE the LLVM/MLIR include_directories() below, so it
# does not inherit them -- it must not depend on the compiler stack. See
# runtime/CMakeLists.txt.
option(HEXIR_BUILD_RUNTIME "Build the standalone hexir runtime" ON)
if(HEXIR_BUILD_RUNTIME)
  set(HEXIR_TOOLS_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  add_subdirectory(runtime)
endif()

include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# Shared by every hexir library and by the driver. See cmake/HexirLibrary.cmake.
set(HEXIR_MLIR_LIBS
          ${dialect_libs}
          ${conversion_libs}
          ${extension_libs}
          LLVMSupport
          MLIRAnalysis
          MLIRBuiltinToLLVMIRTranslation
          MLIRCallInterfaces
          MLIRCastInterfaces
          MLIRExecutionEngine
          MLIRFunctionInterfaces
          MLIRIR
          MLIRLLVMCommonConversion
          MLIRLLVMDialect
          MLIRLLVMToLLVMIRTranslation
          MLIRMemRefDialect
          MLIRParser
          MLIRPass
          MLIRSideEffectInterfaces
          MLIRSupport
          MLIRTargetLLVMIRExport
          MLIRTransforms
          MLIRLinalgDialect
          MLIRTensorDialect
          MLIRArithDialect
          MLIRIR
          MLIRSupport
          MLIRParser
          MLIRAnalysis
          MLIRPass
          MLIRTransforms
          MLIRFuncDialect
          MLIRGPUDialect
          MLIRGPUToLLVMIRTranslation
          MLIRGPUToNVVMTransforms
          MLIRGPUTransforms
          MLIRArithDialect
          MLIRMemRefDialect
          MLIRLinalgDialect
          MLIRAffineDialect
          MLIRSCFDialect
          MLIRLLVMDialect
          MLIRNVVMDialect
          MLIRNVVMToLLVMIRTranslation
          MLIRNVVMToLLVM
          MLIRTensorDialect
          MLIRFuncTransforms
          MLIRLinalgTransforms
          MLIRAffineToStandard
          MLIRSCFToControlFlow
          MLIRArithToLLVM
          MLIRMemRefToLLVM
          MLIRFuncToLLVM
          MLIRLLVMCommonConversion
          MLIRLLVMToLLVMIRTranslation
          MLIRTargetLLVMIRExport
          MLIRBuiltinToLLVMIRTranslation
          MLIRTranslateLib
          MLIRExecutionEngine
          MLIRArithTransforms
          MLIRCallInterfaces
          MLIRBufferizationTransforms
          MLIRCastInterfaces
          MLIRFunctionInterfaces
          MLIRSideEffectInterfaces
          MLIRArithDialect
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRBufferizationDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    
          MLIRRegisterAllExtensions
          MLIRUBToLLVM
          MLIRNVVMTarget
          LLVMSupport)

# include/ generates the TableGen .inc files; compiler/ builds the module libraries
# and registers them in the HEXIR_LIBS global property; tools/ links them.
add_subdirectory(include)
add_subdirectory(compiler)
add_subdirectory(tools)

# Lit test suites. The compiler and the runtime each own their tests:
#   compiler/test  needs only `hexir`
#   runtime/test   runs `hexir-run` on modules `hexir` produced
find_program(LLVM_LIT_EXECUTABLE NAMES llvm-lit lit)
if(LLVM_LIT_EXECUTABLE)
  set(HEXIR_LIT_ARGS -v --param build_dir=${CMAKE_CURRENT_BINARY_DIR})

  add_custom_target(check-hexir-compiler
    COMMAND ${LLVM_LIT_EXECUTABLE} ${HEXIR_LIT_ARGS}
            ${CMAKE_CURRENT_SOURCE_DIR}/compiler/test
    DEPENDS hexir
    COMMENT "Running Hexir compiler tests"
    USES_TERMINAL)

  add_custom_target(check-hexir-runtime
    COMMAND ${LLVM_LIT_EXECUTABLE} ${HEXIR_LIT_ARGS}
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/test
    DEPENDS hexir hexir-run
    COMMENT "Running Hexir runtime tests"
    USES_TERMINAL)

  # Both suites in one lit invocation, so the summary counts everything.
  add_custom_target(check-hexir
    COMMAND ${LLVM_LIT_EXECUTABLE} ${HEXIR_LIT_ARGS}
            ${CMAKE_CURRENT_SOURCE_DIR}/compiler/test
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/test
    DEPENDS hexir hexir-run
    COMMENT "Running all Hexir tests"
    USES_TERMINAL)
else()
  message(WARNING "lit not found: check-hexir targets unavailable (pip install lit)")
endif()
