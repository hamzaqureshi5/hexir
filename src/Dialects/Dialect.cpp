//===- Dialect.cpp - hexir dialect (only add) -----------------------------===//
//
// Minimal hexir dialect implementation with a single AddOp.
//
//===----------------------------------------------------------------------===//

#include "Dialect.h"
#include "Dialects/HexTIRDialect.h"

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
using namespace mlir::hexir;

/// Generated dialect definitions (HexirDialect, etc.).
#include "Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// HexirInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with Hexir
/// operations.
struct HexirInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within hexir can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within hexir can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  // All functions within hexir can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator(hexir.return) by replacing it with a new
  /// operation as necessary.
  // void handleTerminator(Operation *op,
  // ArrayRef<Value> valuesToRepl) const final {
  // // Only "hexir.return" needs to be handled here.
  // auto returnOp = cast<ReturnOp>(op);

  // // Replace the values directly with the return operands.
  // assert(returnOp.getNumOperands() == valuesToRepl.size());
  // for (const auto &it : llvm::enumerate(returnOp.getOperands()))
  // valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  // }

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  // Operation *materializeCallConversion(OpBuilder &builder, Value input,
  // Type resultType,
  // Location conversionLoc) const final {
  // return builder.create<CastOp>(conversionLoc, resultType, input);
  // }
};

/// Generated op method definitions (for AddOp).
#define GET_OP_CLASSES
#include "Ops.cpp.inc"
#undef GET_OP_CLASSES

/// Dialect initialization: register AddOp only.
void HexirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      // #undef GET_OP_LIST
      >();
  // addInterfaces<HexirInlinerInterface>();
  // addTypes<StructType>();


}

// void HexirDialect::initialize() {
// addOperations<
// #define GET_OP_LIST
// #include "Ops.cpp.inc"
// >();
// addInterfaces<HexirInlinerInterface>();
// addTypes<StructType>();
// }

// mlir::Operation *HexirDialect::materializeConstant(mlir::OpBuilder &builder,
//                                                  mlir::Attribute value,
//                                                  mlir::Type type,
//                                                  mlir::Location loc) {
//   return builder.create<ConstantOp>(loc, type,
//                                     llvm::cast<mlir::DenseElementsAttr>(value));
// }

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.

//===----------------------------------------------------------------------===//
// HexirDialect type parsing / printing
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// HexirDialect type parsing / printing
//===----------------------------------------------------------------------===//

mlir::Type HexirDialect::parseType(mlir::DialectAsmParser &parser) const {
  // If you do NOT want custom types, just reject all:
  parser.emitError(parser.getCurrentLocation(),
                   "hexir dialect has no custom types");
  return Type();
}

void HexirDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  // We should never be asked to print a hexir-specific type in this minimal
  // setup.
  llvm_unreachable("hexir dialect has no custom types to print");
}

//===----------------------------------------------------------------------===//
// HexirDialect constant materializer
//===----------------------------------------------------------------------===//

// mlir::Operation *HexirDialect::materializeConstant(mlir::OpBuilder &builder,
// mlir::Attribute value,
// mlir::Type type,
// mlir::Location loc) {
// // If you do not use hexir.constant anymore, just return nullptr.
// return nullptr;
// }

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

namespace mlir {
namespace hexir {

  //===----------------------------------------------------------------------===//
// Helpers for unary ops
//===----------------------------------------------------------------------===//

/// Parse a unary operation: one operand, one result.
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

/// Print a unary operation
static void printUnaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperand(0);
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : " << *op->result_type_begin();
}

/// Parse a unary op with an extra float attribute (for LeakyRelu, ELU, etc.)
static mlir::ParseResult parseUnaryOpWithAttr(mlir::OpAsmParser &parser,
                                              mlir::OperationState &result,
                                              StringRef attrName) {
  mlir::OpAsmParser::UnresolvedOperand operand;
  mlir::Type type;
  mlir::Attribute attr;
  if (parser.parseOperand(operand) || 
      parser.parseOptionalAttrDict(result.attributes) || 
      parser.parseColonType(type) || 
      parser.parseAttribute(attr, attrName, result.attributes))
    return mlir::failure();

  if (parser.resolveOperand(operand, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// Print unary op with extra attribute
static void printUnaryOpWithAttr(mlir::OpAsmPrinter &printer,
                                 mlir::Operation *op, StringRef attrName) {
  printer << " " << op->getOperand(0);
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : " << *op->result_type_begin();
}

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

// / A generalized printer for binary operations. It prints in two different
// / forms depending on if all of the types match.
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

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

// void AddOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value
// rhs) {
//   auto resultType = lhs.getType().cast<RankedTensorType>();
//   state.addOperands({lhs, rhs});
//   state.addTypes(resultType);
// }

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

ParseResult AddOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBinaryOp(parser, result);
}

void AddOp::print(OpAsmPrinter &p) { printBinaryOp(p, *this); }

// void AddOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//

void mlir::hexir::LinearOp::build(OpBuilder &builder, OperationState &state,
                                Value lhs, Value rhs) {

  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

ParseResult LinearOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBinaryOp(parser, result);
}

void LinearOp::print(OpAsmPrinter &p) { printBinaryOp(p, *this); }



//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

void ReluOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value input) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({input});
}

ParseResult ReluOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseUnaryOp(parser, result);
}

void ReluOp::print(OpAsmPrinter &p) { printUnaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// LeakyReluOp
//===----------------------------------------------------------------------===//

void LeakyReluOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value input) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({input});
}

ParseResult LeakyReluOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseUnaryOpWithAttr(parser, result, "alpha");
}

void LeakyReluOp::print(OpAsmPrinter &p) { printUnaryOpWithAttr(p, *this, "alpha"); }

//===----------------------------------------------------------------------===//
// EluOp
//===----------------------------------------------------------------------===//

void EluOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value input) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({input});
}

ParseResult EluOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseUnaryOpWithAttr(parser, result, "alpha");
}

void EluOp::print(OpAsmPrinter &p) { printUnaryOpWithAttr(p, *this, "alpha"); }

//===----------------------------------------------------------------------===//
// SigmoidOp
//===----------------------------------------------------------------------===//

void SigmoidOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      mlir::Value input) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({input});
}

ParseResult SigmoidOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseUnaryOp(parser, result);
}

void SigmoidOp::print(OpAsmPrinter &p) { printUnaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// TanhOp
//===----------------------------------------------------------------------===//

void TanhOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value input) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({input});
}

ParseResult TanhOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseUnaryOp(parser, result);
}

void TanhOp::print(OpAsmPrinter &p) { printUnaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

void SoftmaxOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      mlir::Value input) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({input});
}

ParseResult SoftmaxOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseUnaryOp(parser, result);
}

void SoftmaxOp::print(OpAsmPrinter &p) { printUnaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// GeluOp
//===----------------------------------------------------------------------===//

void GeluOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value input) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({input});
}

ParseResult GeluOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseUnaryOp(parser, result);
}

void GeluOp::print(OpAsmPrinter &p) { printUnaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// SwishOp
//===----------------------------------------------------------------------===//

void SwishOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Value input) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({input});
}

ParseResult SwishOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseUnaryOp(parser, result);
}

void SwishOp::print(OpAsmPrinter &p) { printUnaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// MishOp
//===----------------------------------------------------------------------===//

void MishOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value input) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({input});
}

ParseResult MishOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseUnaryOp(parser, result);
}

void MishOp::print(OpAsmPrinter &p) { printUnaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// CallTIROp
//===----------------------------------------------------------------------===//

llvm::LogicalResult
CallTIROp::verifySymbolUses(mlir::SymbolTableCollection &symbolTable) {
  auto callee = symbolTable.lookupNearestSymbolFrom<mlir::hextir::PrimFuncOp>(
      *this, getCalleeAttr());
  if (!callee)
    return emitOpError() << "'" << getCallee()
                         << "' does not reference a valid hextir.prim_func";

  // A prim func is destination-passing: one buffer per tensor argument, plus
  // one for the result the call materializes.
  size_t expected = getArgs().size() + 1;
  if (callee.getNumArguments() != expected)
    return emitOpError() << "expected @" << getCallee() << " to take "
                         << expected << " buffers (" << getArgs().size()
                         << " inputs + 1 destination), but it takes "
                         << callee.getNumArguments();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
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

void FuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}



//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"value"});
  printer << getValue();
}

// Verify that the given attribute value is valid for the given type.
static mlir::LogicalResult verifyConstantForType(mlir::Type type,
                                                 mlir::Attribute opaqueValue,
                                                 mlir::Operation *op) {
  if (llvm::isa<mlir::TensorType>(type)) {
    // Check that the value is an elements attribute.
    auto attrValue = llvm::dyn_cast<mlir::DenseFPElementsAttr>(opaqueValue);
    if (!attrValue)
      return op->emitError("constant of TensorType must be initialized by a "
                           "DenseFPElementsAttr, got ")
             << opaqueValue;

    // If the return type of the constant is not an unranked tensor, the shape
    // must match the shape of the attribute holding the data.
    auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(type);
    if (!resultType)
      return success();

    // Check that the rank of the attribute type matches the rank of the
    // constant result type.
    auto attrType = llvm::cast<mlir::RankedTensorType>(attrValue.getType());
    if (attrType.getRank() != resultType.getRank()) {
      return op->emitOpError("return type must match the one of the attached "
                             "value attribute: ")
             << attrType.getRank() << " != " << resultType.getRank();
    }

    // Check that each of the dimensions match between the two types.
    for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
      if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
        return op->emitOpError(
                   "return type shape mismatches its attribute at dimension")
               << dim << ": " << attrType.getShape()[dim]
               << " != " << resultType.getShape()[dim];
      }
    }
    return mlir::success();
  }

  //  auto resultType = llvm::cast<StructType>(type);
  //  llvm::ArrayRef<mlir::Type> resultElementTypes =
  //  resultType.getElementTypes();

  // Verify that the initializer is an Array.
  // auto attrValue = llvm::dyn_cast<ArrayAttr>(opaqueValue);
  // if (!attrValue || attrValue.getValue().size() != resultElementTypes.size())
  //   return op->emitError("constant of StructType must be initialized by an "
  //                        "ArrayAttr with the same number of elements, got ")
  //                        << opaqueValue;

  // Check that each of the elements are valid.
  // llvm::ArrayRef<mlir::Attribute> attrElementValues = attrValue.getValue();
  // for (const auto it : llvm::zip(resultElementTypes, attrElementValues))
  //   if (failed(verifyConstantForType(std::get<0>(it), std::get<1>(it), op)))
  //     return mlir::failure();
  // return mlir::success();
}

// Verifier for the constant operation. This corresponds to the `::verify(...)`
// in the op definition.
mlir::LogicalResult ConstantOp::verify() {
  return verifyConstantForType(getResult().getType(), getValue(), *this);
}


//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

// /// Infer the output shape of the CastOp, this is required by the shape
// /// inference interface.
// void CastOp::inferShapes() { getResult().setType(getInput().getType()); }

// /// Returns true if the given set of input and result types are compatible
// with
// /// this cast operation. This is required by the `CastOpInterface` to verify
// /// this operation and provide other additional utilities.
// bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
// if (inputs.size() != 1 || outputs.size() != 1)
// return false;
// // The inputs must be Tensors with the same element type.
// TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
// TensorType output = llvm::dyn_cast<TensorType>(outputs.front());
// if (!input || !output || input.getElementType() != output.getElementType())
// return false;
// // The shape is required to match if both types are ranked.
// return !input.hasRank() || !output.hasRank() || input == output;
// }

} // namespace hexir
} // namespace mlir

//===----------------------------------------------------------------------===//
// hexir Types
//===----------------------------------------------------------------------===//

// namespace mlir {
// namespace hexir {
// namespace detail {
// /// This class represents the internal storage of the hexir `StructType`.
// struct StructTypeStorage : public mlir::TypeStorage {
//   /// The `KeyTy` is a required type that provides an interface for the
//   storage
//   /// instance. This type will be used when uniquing an instance of the type
//   /// storage. For our struct type, we will unique each instance structurally
//   on
//   /// the elements that it contains.
//   using KeyTy = llvm::ArrayRef<mlir::Type>;

//   /// A constructor for the type storage instance.
//   StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
//       : elementTypes(elementTypes) {}

//   /// Define the comparison function for the key type with the current
//   storage
//   /// instance. This is used when constructing a new instance to ensure that
//   we
//   /// haven't already uniqued an instance of the given key.
//   bool operator==(const KeyTy &key) const { return key == elementTypes; }

//   /// Define a hash function for the key type. This is used when uniquing
//   /// instances of the storage, see the `StructType::get` method.
//   /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
//   /// have hash functions available, so we could just omit this entirely.
//   static llvm::hash_code hashKey(const KeyTy &key) {
//     return llvm::hash_value(key);
//   }

//   /// Define a construction function for the key type from a set of
//   parameters.
//   /// These parameters will be provided when constructing the storage
//   instance
//   /// itself.
//   /// Note: This method isn't necessary because KeyTy can be directly
//   /// constructed with the given parameters.
//   static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
//     return KeyTy(elementTypes);
//   }

//   /// Define a construction method for creating a new instance of this
//   storage.
//   /// This method takes an instance of a storage allocator, and an instance
//   of a
//   /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
//   /// allocations used to create the type storage and its internal.
//   static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
//                                       const KeyTy &key) {
//     // Copy the elements from the provided `KeyTy` into the allocator.
//     llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

//     // Allocate the storage instance and construct it.
//     return new (allocator.allocate<StructTypeStorage>())
//         StructTypeStorage(elementTypes);
//   }

//   /// The following field contains the element types of the struct.
//   llvm::ArrayRef<mlir::Type> elementTypes;
// };
// } // namespace detail
// } // namespace hexir
// } // namespace mlir

/// Create an instance of a `StructType` with the given element types. There
/// *must* be at least one element type.
// StructType StructType::get(llvm::ArrayRef<mlir::Type> elementTypes) {
//   assert(!elementTypes.empty() && "expected at least 1 element type");

//   // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
//   // of this type. The first parameter is the context to unique in. The
//   // parameters after the context are forwarded to the storage instance.
//   mlir::MLIRContext *ctx = elementTypes.front().getContext();
//   return Base::get(ctx, elementTypes);
// }

/// Returns the element types of this struct type.
// llvm::ArrayRef<mlir::Type> StructType::getElementTypes() {
//   // 'getImpl' returns a pointer to the internal storage instance.
//   return getImpl()->elementTypes;
// }

/// Parse an instance of a type registered to the hexir dialect.
// mlir::Type HexirDialect::parseType(mlir::DialectAsmParser &parser) const {
// // Parse a struct type in the following form:
// // struct-type ::= `struct` `<` type (`,` type)* `>`

// // NOTE: All MLIR parser function return a ParseResult. This is a
// // specialization of LogicalResult that auto-converts to a `true` boolean
// // value on failure to allow for chaining, but may be used with explicit
// // `mlir::failed/mlir::succeeded` as desired.

// // Parse: `struct` `<`
// if (parser.parseKeyword("struct") || parser.parseLess())
// return Type();

// // Parse the element types of the struct.
// SmallVector<mlir::Type, 1> elementTypes;
// do {
// // Parse the current element type.
// SMLoc typeLoc = parser.getCurrentLocation();
// mlir::Type elementType;
// if (parser.parseType(elementType))
// return nullptr;

// // Check that the type is either a TensorType or another StructType.
// if (!llvm::isa<mlir::TensorType, StructType>(elementType)) {
// parser.emitError(typeLoc, "element type for a struct must either "
// "be a TensorType or a StructType, got: ")
// << elementType;
// return Type();
// }
// elementTypes.push_back(elementType);

// // Parse the optional: `,`
// } while (succeeded(parser.parseOptionalComma()));

// // Parse: `>`
// if (parser.parseGreater())
// return Type();
// return StructType::get(elementTypes);
// }