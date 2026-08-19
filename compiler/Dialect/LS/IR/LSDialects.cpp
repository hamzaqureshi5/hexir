//===- LSDialects.cpp - Local-system CPU/GPU dialects ----------*- C++ -*-===//

#include "hexir/Dialect/LS/IR/LSDialects.h"
#include "hexir/Dialect/Hexir/IR/HexirDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

using namespace mlir;

#include "hexir/Dialect/LS/IR/LSCPUDialect.cpp.inc"
#include "hexir/Dialect/LS/IR/LSGPUDialect.cpp.inc"

#define GET_OP_CLASSES
#include "hexir/Dialect/LS/IR/LSDialectsOps.cpp.inc"

static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {

  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

static mlir::ParseResult parseUnaryOp(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand operand;
  mlir::Type type;
  if (parser.parseOperand(operand) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  if (parser.resolveOperand(operand, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

static void printUnaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperand(0);
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : " << *op->result_type_begin();
}

void ls_cpu::LSCPUDialect::initialize() {
  addOperations<ls_cpu::AddOp, ls_cpu::MulOp, ls_cpu::MatmulOp,
                ls_cpu::ReluOp>();
}

void mlir::ls_cpu::AddOp::build(OpBuilder &builder, OperationState &state,
                                Value lhs, Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

ParseResult ls_cpu::AddOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBinaryOp(parser, result);
}

void ls_cpu::AddOp::print(OpAsmPrinter &p) { printBinaryOp(p, *this); }

void mlir::ls_cpu::MulOp::build(OpBuilder &builder, OperationState &state,
                                Value lhs, Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

ParseResult ls_cpu::MulOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBinaryOp(parser, result);
}

void ls_cpu::MulOp::print(OpAsmPrinter &p) { printBinaryOp(p, *this); }

void mlir::ls_cpu::MatmulOp::build(OpBuilder &builder, OperationState &state,
                                   Value lhs, Value rhs) {

  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

ParseResult ls_cpu::MatmulOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  return parseBinaryOp(parser, result);
}

void ls_cpu::MatmulOp::print(OpAsmPrinter &p) { printBinaryOp(p, *this); }

void mlir::ls_cpu::ReluOp::build(OpBuilder &builder, OperationState &state,
                                 Value input) {
  state.addTypes(input.getType());
  state.addOperands(input);
}

ParseResult ls_cpu::ReluOp::parse(OpAsmParser &parser,
                                  OperationState &result) {
  return parseUnaryOp(parser, result);
}

void ls_cpu::ReluOp::print(OpAsmPrinter &p) { printUnaryOp(p, *this); }

void ls_gpu::LSGPUDialect::initialize() {
  addOperations<ls_gpu::AddOp, ls_gpu::MulOp, ls_gpu::MatmulOp,
                ls_gpu::ReluOp>();
}

void mlir::ls_gpu::AddOp::build(OpBuilder &builder, OperationState &state,
                                Value lhs, Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

ParseResult ls_gpu::AddOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBinaryOp(parser, result);
}

void ls_gpu::AddOp::print(OpAsmPrinter &p) { printBinaryOp(p, *this); }

void mlir::ls_gpu::MulOp::build(OpBuilder &builder, OperationState &state,
                                Value lhs, Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

ParseResult ls_gpu::MulOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBinaryOp(parser, result);
}

void ls_gpu::MulOp::print(OpAsmPrinter &p) { printBinaryOp(p, *this); }

void mlir::ls_gpu::MatmulOp::build(OpBuilder &builder, OperationState &state,
                                   Value lhs, Value rhs) {

  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

ParseResult ls_gpu::MatmulOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  return parseBinaryOp(parser, result);
}

void ls_gpu::MatmulOp::print(OpAsmPrinter &p) { printBinaryOp(p, *this); }

void mlir::ls_gpu::ReluOp::build(OpBuilder &builder, OperationState &state,
                                 Value input) {
  state.addTypes(input.getType());
  state.addOperands(input);
}

ParseResult ls_gpu::ReluOp::parse(OpAsmParser &parser,
                                  OperationState &result) {
  return parseUnaryOp(parser, result);
}

void ls_gpu::ReluOp::print(OpAsmPrinter &p) { printUnaryOp(p, *this); }
