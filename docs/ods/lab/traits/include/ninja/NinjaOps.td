//===- NinjaOps.td - A small dialect for learning ODS -------*- tablegen -*-===//
//
// `ninja` exists to learn TableGen and
// ODS (Operation Definition Specification) on: every op below shows one idea,
// and nothing lowers it, so it can be changed freely.
//
// Try it:   ./build/ninja-opt test/ninja-roundtrip.mlir
// Generated C++ lands in build/include/ninja/*.inc
//
//===----------------------------------------------------------------------===//

#ifndef NINJA_OPS
#define NINJA_OPS

// OpBase.td defines Dialect, Op, the type constraints (F64, AnyFloat...) and
// the attribute constraints (F64Attr, I32Attr...).
include "mlir/IR/OpBase.td"
// Pure (= no side effects + speculatable) comes from here.
include "mlir/Interfaces/SideEffectInterfaces.td"
// SameOperandsAndResultType lives here, because it also infers result types.
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// The dialect
//===----------------------------------------------------------------------===//

// `-gen-dialect-decls` turns this into the C++ class `NinjaDialect`.
def Ninja_Dialect : Dialect {
  let name = "ninja";                    // the prefix in `ninja.add`
  let cppNamespace = "::mlir::ninja";
  let summary = "a toy dialect for learning ODS";
}

// A base class so every op does not have to repeat `Ninja_Dialect`.
// `mnemonic` is the part after the dot; `traits` defaults to none.
class Ninja_Op<string mnemonic, list<Trait> traits = []>
    : Op<Ninja_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Lesson 1: an attribute in, a value out
//===----------------------------------------------------------------------===//

def Ninja_ConstantOp : Ninja_Op<"constant", [Pure]> {
  let summary = "a constant f64";
  let description = [{
    ```mlir
    %x = ninja.constant 2.5
    ```
  }];

  // An attribute is compile-time data stored on the op; it is not an SSA value.
  let arguments = (ins F64Attr:$value);
  let results = (outs F64:$result);

  // Declarative assembly format: ODS writes the parser and printer for you.
  // The result type is fixed (F64), so it does not need to be spelled out.
  let assemblyFormat = "$value attr-dict";
}

//===----------------------------------------------------------------------===//
// Lesson 2: operands and traits
//===----------------------------------------------------------------------===//

def Ninja_AddOp : Ninja_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "add two floats";
  let description = [{
    ```mlir
    %z = ninja.add %x, %y : f64
    ```
  }];

  // Operands are SSA values. AnyFloat is a type constraint: the generated
  // verifier rejects an i32 here without any C++ from us.
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs);
  let results = (outs AnyFloat:$result);

  // SameOperandsAndResultType lets the format name one type and infer the rest.
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Lesson 3: a custom verifier
//===----------------------------------------------------------------------===//

def Ninja_StrikeOp : Ninja_Op<"strike", [Pure]> {
  let summary = "strike a target some number of times";
  let description = [{
    Multiplies `target` by `times`. `times` must be positive -- a rule no
    type constraint can express, so it lives in a C++ verifier.

    ```mlir
    %y = ninja.strike %x times 3
    ```
  }];

  let arguments = (ins F64:$target, I32Attr:$times);
  let results = (outs F64:$result);

  // `times` in backticks is a literal keyword; $times is the attribute.
//let assemblyFormat = "$target `times` $times attr-dict";
  //let assemblyFormat = "$target `times` $times attr-dict  type($result)";
  let assemblyFormat = "$target `:` type($target) `times` $times attr-dict `:` type($result)";
  // Declares `LogicalResult verify();` -- we write the body in NinjaDialect.cpp.
  let hasVerifier = 1;
}

include "ninja/TraitsLab.td"

#endif // NINJA_OPS
