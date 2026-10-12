//===- ninja-opt.cpp - mlir-opt with the ninja dialect ---------*- C++ -*-===//
//
// Parses a file, verifies it, runs any passes given on the command line, and
// prints it back. Everything mlir-opt can do, plus the ninja dialect.
//
//   ./build/ninja-opt test/ninja-roundtrip.mlir
//   ./build/ninja-opt --canonicalize test/ninja-roundtrip.mlir
//
//===----------------------------------------------------------------------===//

#include "ninja/NinjaDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerTransformsPasses(); // canonicalize, cse, ...

  mlir::DialectRegistry registry;
  registry.insert<mlir::ninja::NinjaDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ninja optimizer driver\n", registry));
}
