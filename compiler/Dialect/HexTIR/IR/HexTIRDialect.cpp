//===- HexTIRDialect.cpp - Hexir kernel-level dialect ----------*- C++ -*-===//

#include "hexir/Dialect/HexTIR/IR/HexTIRDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::hextir;

#include "hexir/Dialect/HexTIR/IR/HexTIRDialect.cpp.inc"

#define GET_OP_CLASSES
#include "hexir/Dialect/HexTIR/IR/HexTIROps.cpp.inc"

//===----------------------------------------------------------------------===//
// HexTIRDialect
//===----------------------------------------------------------------------===//

void HexTIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hexir/Dialect/HexTIR/IR/HexTIROps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// PrimFuncOp
//===----------------------------------------------------------------------===//

void PrimFuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       llvm::StringRef name, mlir::FunctionType type,
                       llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface gives us a builder that populates the state and creates
  // the entry block with one argument per buffer parameter.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult PrimFuncOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void PrimFuncOp::print(mlir::OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

void ForOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lowerBound, mlir::Value upperBound,
                  mlir::Value step, llvm::StringRef kind,
                  mlir::ValueRange initArgs) {
  state.addOperands({lowerBound, upperBound, step});
  state.addOperands(initArgs);
  state.addAttribute(getKindAttrName(state.name), builder.getStringAttr(kind));
  // One result per loop-carried value, matching scf.for.
  for (Value init : initArgs)
    state.addTypes(init.getType());

  // The body takes the induction variable first, then one argument per
  // loop-carried value, and is terminated by hextir.yield.
  Region *body = state.addRegion();
  Block &block = body->emplaceBlock();
  block.addArgument(builder.getIndexType(), state.location);
  for (Value init : initArgs)
    block.addArgument(init.getType(), state.location);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&block);
  // The terminator has to yield the loop-carried values back; a caller that
  // adds iter_args replaces this with a yield of the updated values.
  YieldOp::create(builder, state.location, block.getArguments().drop_front());
}
