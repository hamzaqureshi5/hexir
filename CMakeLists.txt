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

# include/ generates the TableGen .inc files; lib/ builds the module libraries
# and registers them in the HEXIR_LIBS global property; tools/ links them.
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# Lit test suite: `make check-hexir` runs FileCheck tests in test/.
find_program(LLVM_LIT_EXECUTABLE NAMES llvm-lit lit)
if(LLVM_LIT_EXECUTABLE)
  add_custom_target(check-hexir
    COMMAND ${LLVM_LIT_EXECUTABLE} -v ${CMAKE_CURRENT_SOURCE_DIR}/test
            --param build_dir=${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS hexir
    COMMENT "Running Hexir lit tests"
    USES_TERMINAL)
else()
  message(WARNING "lit not found: check-hexir target unavailable (pip install lit)")
endif()
