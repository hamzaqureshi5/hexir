//===- NinjaDialect.cpp - A small dialect for learning ODS -----*- C++ -*-===//

#include "ninja/NinjaDialect.h"

using namespace mlir;
using namespace mlir::ninja;

#include "ninja/NinjaDialect.cpp.inc"

#define GET_OP_CLASSES
#include "ninja/NinjaOps.cpp.inc"

//===----------------------------------------------------------------------===//
// NinjaDialect
//===----------------------------------------------------------------------===//

void NinjaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ninja/NinjaOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// StrikeOp
//===----------------------------------------------------------------------===//

// Runs after the ODS-generated checks (operand types, attribute present and
// i32), so getTimes() is safe to call here.
LogicalResult StrikeOp::verify() {
  if (getTimes() <= 0)
    return emitOpError("times must be positive, got ") << getTimes();
  return success();
}
