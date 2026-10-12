#ifndef NINJA_OPS
#define NINJA_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def Ninja_Dialect : Dialect {
  let name = "ninja";
  let cppNamespace = "::mlir::ninja";
  let summary = "a toy dialect for learning ODS";
  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class Ninja_Op<string mnemonic, list<Trait> traits = []>
    : Op<Ninja_Dialect, mnemonic, traits>;

// ---- Types (Tier 3)
class Ninja_Type<string name, string typeMnemonic> : TypeDef<Ninja_Dialect, name> {
  let mnemonic = typeMnemonic;
}
def Ninja_ShurikenType : Ninja_Type<"Shuriken", "shuriken"> {
  let summary = "a throwing star with N points";
  let parameters = (ins "unsigned":$points);
  let assemblyFormat = "`<` $points `>`";
  let genVerifyDecl = 1;
}

// ---- Attribute
def Ninja_RankAttr : AttrDef<Ninja_Dialect, "Rank"> {
  let mnemonic = "rank";
  let summary = "a ninja's rank: a level and a title";
  let parameters = (ins "unsigned":$level, StringRefParameter<"the title">:$title);
  let assemblyFormat = "`<` $level `,` $title `>`";
}

def Ninja_PromoteOp : Ninja_Op<"promote"> {
  let arguments = (ins Ninja_RankAttr:$rank);
  let assemblyFormat = "$rank attr-dict";
}

// ---- Enum (Tier 3)
def Ninja_Stealth : I32EnumAttrCase<"stealth", 0>;
def Ninja_Attack  : I32EnumAttrCase<"attack", 1>;
def Ninja_StanceAttr : I32EnumAttr<"Stance", "a ninja's stance",
    [Ninja_Stealth, Ninja_Attack]> {
  let cppNamespace = "::mlir::ninja";
}

// ---- Original ops, plus folding / canonicalization (Tier 2)
def Ninja_ConstantOp : Ninja_Op<"constant", [Pure, ConstantLike]> {
  let summary = "a constant f64";
  let description = [{
    Makes an `f64` value from a number written on the op.

    ```mlir
    %x = ninja.constant 2.5
    ```
  }];
  let arguments = (ins F64Attr:$value);
  let results = (outs F64:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
  let builders = [
    OpBuilder<(ins "double":$v), [{
      build($_builder, $_state, $_builder.getF64FloatAttr(v));
    }]>
  ];
}

def Ninja_AddOp : Ninja_Op<"add", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "add two floats";
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs);
  let results = (outs AnyFloat:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Ninja_StrikeOp : Ninja_Op<"strike", [Pure]> {
  let summary = "strike a target some number of times";
  let arguments = (ins F64:$target, I32Attr:$times);
  let results = (outs F64:$result);
  let assemblyFormat = "$target `:` type($target) `times` $times attr-dict `:` type($result)";
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
  let extraClassDeclaration = [{
    /// True when this strike hits only once.
    bool isSingleStrike() { return getTimes() == 1; }
  }];
}

// ---- Attribute constraints + optional groups (sections 6, 8)
def Ninja_DodgeOp : Ninja_Op<"dodge"> {
  let summary = "dodge in a direction";
  let arguments = (ins
    StrAttr:$direction,
    OptionalAttr<I32Attr>:$steps,
    UnitAttr:$fast,
    DefaultValuedAttr<F64Attr, "1.0">:$speed
  );
  let assemblyFormat = "$direction (`steps` $steps^)? (`fast` $fast^)? (`speed` $speed^)? attr-dict";
}

def Ninja_ShadowOp : Ninja_Op<"shadow"> {
  let summary = "make shadow clones";
  let arguments = (ins ConfinedAttr<I32Attr, [IntPositive]>:$count);
  let assemblyFormat = "$count attr-dict";
}

// ---- Variadic, functional-type (sections 5, 8)
def Ninja_SumOp : Ninja_Op<"sum", [Pure]> {
  let summary = "add any number of f64 values";
  let arguments = (ins Variadic<F64>:$inputs);
  let results = (outs F64:$result);
  let assemblyFormat = "$inputs attr-dict";
}

def Ninja_MixOp : Ninja_Op<"mix"> {
  let summary = "anything in, anything out";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $result)";
}

// ---- Enum + custom type in use
def Ninja_StanceOp : Ninja_Op<"stance"> {
  let summary = "take a stance";
  let arguments = (ins Ninja_StanceAttr:$stance);
  let assemblyFormat = "$stance attr-dict";
}

def Ninja_ThrowOp : Ninja_Op<"throw"> {
  let summary = "throw a shuriken";
  let arguments = (ins Ninja_ShurikenType:$star);
  let assemblyFormat = "$star attr-dict `:` qualified(type($star))";
}

def Ninja_ForgeOp : Ninja_Op<"forge"> {
  let summary = "make a shuriken";
  let results = (outs Ninja_ShurikenType:$star);
  let assemblyFormat = "attr-dict `:` type($star)";
}

// ---- Regions, terminators, HasParent (sections 7, 17)
def Ninja_DojoOp : Ninja_Op<"dojo", [
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "a training hall: a region of ops";
  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$name attr-dict-with-keyword $body";
}

def Ninja_YieldOp : Ninja_Op<"yield", [Pure, Terminator, HasParent<"DojoOp">]> {
  let summary = "end of a dojo";
  let assemblyFormat = "attr-dict";
}

def Ninja_StarsOp : Ninja_Op<"stars"> {
  let arguments = (ins I32Attr:$count);
  let assemblyFormat = "custom<Stars>($count) attr-dict";
}

#endif // NINJA_OPS
