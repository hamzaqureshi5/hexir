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

// GraphOp: its single region is a graph region.
RegionKind GraphOp::getRegionKind(unsigned index) { return RegionKind::Graph; }

// ElemOp: the result type is the element type of the input tensor.
LogicalResult ElemOp::inferReturnTypes(MLIRContext *, std::optional<Location>,
                                       ElemOp::Adaptor adaptor,
                                       SmallVectorImpl<Type> &inferred) {
  inferred.push_back(
      cast<ShapedType>(adaptor.getInput().getType()).getElementType());
  return success();
}

// NamedOp: print the result as %sum instead of %0.
void NamedOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "sum");
}

// CastOp: only float-to-float casts are allowed.
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return inputs.size() == 1 && outputs.size() == 1 &&
         isa<FloatType>(inputs.front()) && isa<FloatType>(outputs.front());
}

// ArenaOp: the arena must not be empty.
LogicalResult ArenaOp::verifyRegions() {
  if (getBody().front().empty())
    return emitOpError("arena must contain at least one op");
  return success();
}

// ZipListsOp: SameVariadicOperandSize does not check that the operands split
// evenly; getXs() and getYs() are both computed as half of the total.
LogicalResult ZipListsOp::verify() {
  if ((*this)->getNumOperands() % 2 != 0)
    return emitOpError("expects the same number of xs and ys, got ")
           << (*this)->getNumOperands() << " operands in total";
  return success();
}

// JumpOp (ninja.jump_to): its `args` are passed to the destination block.
SuccessorOperands JumpOp::getSuccessorOperands(unsigned index) {
  return SuccessorOperands(getArgsMutable());
}
