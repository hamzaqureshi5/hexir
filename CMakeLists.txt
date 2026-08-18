cmake_minimum_required(VERSION 3.13.4)
project(hexir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)



set(LLVM_DIR "/home/user/llvm-project/build/lib/cmake/llvm")
set(MLIR_DIR "/home/user/llvm-project/build/lib/cmake/mlir")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)
include(TableGen)
include(HandleLLVMOptions)

set(LLVM_LINK_COMPONENTS Core Support nativecodegen OrcJIT)

include_directories(include)

include_directories(
  ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Dialects
  ${CMAKE_CURRENT_BINARY_DIR}/include # For TableGen-generated .inc files
  ${CMAKE_CURRENT_BINARY_DIR}/include/Dialects # For TableGen-generated .inc files
  ${CMAKE_CURRENT_BINARY_DIR})

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)



set(LLVM_TARGET_DEFINITIONS src/HexirCombine.td)
mlir_tablegen(HexirCombine.inc -gen-rewriters)
add_public_tablegen_target(CombineIncGen)


file(
  GLOB CCMLIR_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
                                  "${CMAKE_CURRENT_SOURCE_DIR}/src/Dialects/*.cpp"
  # GLOB CCMLIR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/targets/cpu/*.cpp"
  # GLOB CCMLIR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/targets/gpu/*.cpp"
  # GLOB CCMLIR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/targets/metal/*.cpp"
  # GLOB CCMLIR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/targets/cpu/*.cpp"
  # GLOB CCMLIR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/targets/rocm/*.cpp"
)
add_executable(hexir ${CCMLIR_SRC})


# add_dependencies(hexir Ops ShapeInfer CombineIncGen)
add_dependencies(hexir OpsIncGen LSDialectsIncGen HexTIRIncGen
                 ShapeInferenceInterfaceIncGen CombineIncGen)

llvm_update_compile_flags(hexir)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(
  hexir
  PRIVATE ${dialect_libs}
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

install(TARGETS hexir RUNTIME DESTINATION bin)

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
