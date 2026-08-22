cmake_minimum_required(VERSION 3.13.4)
project(hexir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Which LLVM to build against, in order of preference:
#
#   1. -DLLVM_DIR=... -DMLIR_DIR=... on the command line
#   2. the pinned submodule, once scripts/build-llvm.sh has built it
#   3. a machine-wide checkout, which is where this project started
#
# The submodule exists because "which LLVM?" is not a detail here: hexir uses
# the new-style OpTy::create(builder, ...) builders, which are in no release, so
# no packaged MLIR can build it. third_party/llvm-project pins the exact commit
# it is known to work with. See third_party/README.md.
set(HEXIR_LLVM_SUBMODULE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/build")
if(NOT DEFINED LLVM_DIR)
  if(EXISTS "${HEXIR_LLVM_SUBMODULE}/lib/cmake/llvm/LLVMConfig.cmake")
    set(LLVM_DIR "${HEXIR_LLVM_SUBMODULE}/lib/cmake/llvm")
  else()
    set(LLVM_DIR "/home/user/llvm-project/build/lib/cmake/llvm")
  endif()
endif()
if(NOT DEFINED MLIR_DIR)
  if(EXISTS "${HEXIR_LLVM_SUBMODULE}/lib/cmake/mlir/MLIRConfig.cmake")
    set(MLIR_DIR "${HEXIR_LLVM_SUBMODULE}/lib/cmake/mlir")
  else()
    set(MLIR_DIR "/home/user/llvm-project/build/lib/cmake/mlir")
  endif()
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# Say which one was picked. Pointing LLVM_DIR at a directory with no config in
# it does not fail -- CMake quietly searches its default prefixes instead, and
# this machine has an installed LLVM in /usr/local as well as two build trees.
# So the config file is what gets tested above, and the choice gets printed.
message(STATUS "hexir: LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")

# The major version the pinned commit carries. A mismatch is worth naming here,
# because the symptom otherwise is pages of template errors from a builder
# signature that changed.
set(HEXIR_LLVM_REQUIRED_MAJOR 24)
if(LLVM_VERSION_MAJOR LESS HEXIR_LLVM_REQUIRED_MAJOR)
  message(FATAL_ERROR
    "hexir needs LLVM ${HEXIR_LLVM_REQUIRED_MAJOR} or newer, found "
    "${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}.\n"
    "The new-style OpTy::create(builder, ...) builders are not in any release, "
    "so a packaged MLIR cannot build this. Build the pinned commit:\n"
    "    git submodule update --init third_party/llvm-project\n"
    "    scripts/build-llvm.sh")
elseif(LLVM_VERSION_MAJOR GREATER HEXIR_LLVM_REQUIRED_MAJOR)
  message(WARNING
    "hexir is pinned to LLVM ${HEXIR_LLVM_REQUIRED_MAJOR} but found "
    "${LLVM_PACKAGE_VERSION}. Probably fine; MLIR does break APIs between "
    "majors, so if this fails to build, that is why.")
endif()

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
