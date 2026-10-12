# Solutions

Try each exercise first. Every answer below was checked by building it and running it, except
where an answer is an explanation.

---

## Chapter 1

**1.1** Record name: `Ninja_StrikeOp`. Class it is built from: `Ninja_Op`. Template arguments:
`"strike"` and `[Pure]`.

**1.2** `string opName = "strike";`. `Ninja_Op` passes its `mnemonic` argument (`"strike"`) on to
MLIR's `Op` class, which stores it in `opName`.

**1.3** `error: Variable not defined: 'Pure'`. `Pure` is defined in
`SideEffectInterfaces.td`, so without that `include` TableGen has never heard of it.

**1.4**

```tablegen
class Ninja<string n, int rank> {
  string name = n;
  int level = rank;
}
def Kenji : Ninja<"kenji", 3>;
def Aiko : Ninja<"aiko", 5>;
```

```
def Aiko {	// Ninja
  string name = "aiko";
  int level = 5;
}
def Kenji {	// Ninja
  string name = "kenji";
  int level = 3;
}
```

---

## Chapter 2

**2.1** In `%y = ninja.strike %x : f64 times 3 : f64`: `%x` is an operand, `3` is an attribute,
`%y` is the result. `f64` is the type of `%x` and of `%y`, printed by the format.

**2.2** The real line is:

```cpp
class StrikeOp : public ::mlir::Op<StrikeOp, ::mlir::OpTrait::ZeroRegions,
    ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::FloatType>::Impl,
    ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants,
    ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait,
    ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait,
    ::mlir::InferTypeOpInterface::Trait>
```

You wrote only `Pure`, which became `ConditionallySpeculatable`, `AlwaysSpeculatableImplTrait`
and `MemoryEffectOpInterface`. ODS added the rest: `ZeroRegions`, `OneResult`,
`OneTypedResult<FloatType>`, `ZeroSuccessors`, `OneOperand` (the shape), `OpInvariants` (always),
`BytecodeOpInterface` (the op has attributes) and `InferTypeOpInterface` (the result type `F64` is
known).

**2.3** `error: Dialect 'ninja' not found for custom op 'ninja.constant'`. It names
`ninja.constant` because that is the first ninja op in the file.

**2.4** (a) An attribute: the value is fixed when the code is written, and can be checked while
compiling (for example "must be positive"). (b) An operand: the value only exists when the
program runs, so it has to arrive on a wire from another op.

---

## Chapter 3

**3.1** The test adds two `f64` values, so the verifier rejects the op:

```
test/ninja-roundtrip.mlir:10:10: error: 'ninja.add' op operand #0 must be 32-bit float, but got 'f64'
```

**3.2** (a) `TensorRankOf<[F32], [2]>`. (b) `AnyTypeOf<[I32, I64]>`. (c) `Variadic<Index>`.

**3.3**

```tablegen
def Ninja_SumOp : Ninja_Op<"sum", [Pure]> {
  let summary = "add any number of f64 values";
  let arguments = (ins Variadic<F64>:$inputs);
  let results = (outs F64:$result);
  let assemblyFormat = "$inputs attr-dict";
}
```

Test (`test/ninja-sum.mlir`):

```mlir
// RUN: %ninja-opt %s | FileCheck %s

func.func @f(%a: f64, %b: f64, %c: f64) -> f64 {
  // CHECK: ninja.sum %{{.*}}, %{{.*}}, %{{.*}}
  %s = ninja.sum %a, %b, %c
  return %s : f64
}
```

**3.4** `3`.

---

## Chapter 4

**4.1** It is accepted and printed back unchanged. In `NinjaOps.h.inc`: `uint32_t getTimes();`.
`-1` becomes 4294967295, which is not `<= 0`.

**4.2**

```tablegen
let arguments = (ins F64:$target, ConfinedAttr<I32Attr, [IntPositive]>:$times);
// and delete:  let hasVerifier = 1;
```

Delete `StrikeOp::verify()` from `lib/NinjaDialect.cpp`. Change the `CHECK` line in
`test/ninja-invalid.mlir` to:

```mlir
  // CHECK: ninja-invalid.mlir:[[@LINE+1]]:8: error: 'ninja.strike' op attribute 'times' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive
```

Results (tested on your current `strike` format):

```
times -1: error: 'ninja.strike' op attribute 'times' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive
times 0:  same error
times 3:  ok
```

**4.3** See chapter 4, section 4.4. The generic form of the last line is:

```mlir
"ninja.dodge"() <{direction = "down", fast, speed = 4.000000e+00 : f64, steps = 2 : i32}> : () -> ()
```

**4.4**

(a) `OptionalAttr<DenseI64ArrayAttr>:$sizes`.

(b) There is no ready-made "non-empty string" rule, so write one:

```tablegen
ConfinedAttr<StrAttr, [AttrConstraint<
    CPred<"!::llvm::cast<::mlir::StringAttr>($_self).getValue().empty()">,
    "whose value is not empty">]>:$name
```

```
error: 'ninja.name_it' op attribute 'name' failed to satisfy constraint: string attribute whose value is not empty
```

(c) `DefaultValuedAttr<F64Attr, "0.01">:$learning_rate`.

---

## Chapter 5

**5.1** The `: f64` after `%x` is gone, so every `ninja.strike` line must be written as
`ninja.strike %x times 3 : f64`. Until you update them, both test files fail with
`custom op 'ninja.strike' expected 'times'`.

**5.2**

```tablegen
def Ninja_JumpOp : Ninja_Op<"jump", [SameTypeOperands]> {
  let arguments = (ins AnyType:$x, AnyType:$y);
  let assemblyFormat = "$x `to` $y attr-dict `:` type($x)";
}
```

`SameTypeOperands` tells the parser that `$y` has the same type as `$x`, so one type is enough.
Tested: `ninja.jump %arg0 to %arg1 : f32`.

**5.3** Without `speed` in the format:

```
error: strict properties in assembly format requires prop-dict unless all inherent attributes
and properties are bound in the custom assembly format; missing attribute 'speed'
```

With `"$direction prop-dict attr-dict"`:

```mlir
ninja.dodge "down" <steps = 2, fast>
ninja.dodge "up"
ninja.dodge "back" <speed = 3.000000e+00>
```

**5.4** Remove the `if (n == 0) return parser.emitError(...)` lines from `parseStars`. Then
`ninja.stars` (no stars) parses, stores `count = 0`, and prints as `ninja.stars` with nothing
after it.

---

## Chapter 6

**6.1** `HasParent<"DojoOp">` (structural trait) → "operand #0 must be 64-bit float" (ODS
checks) → your `verify()` → your `verifyRegions()` (phase 2).

**6.2**

```tablegen
let hasVerifier = 1;      // in Ninja_AddOp
```

```cpp
LogicalResult AddOp::verify() {
  if (getLhs() == getRhs())
    return emitOpError("operands must be different values");
  return success();
}
```

```
ninja-roundtrip.mlir:10:10: error: 'ninja.add' op operands must be different values
```

Your own roundtrip test uses `ninja.add %x, %x`, so it now fails. That is why the exercise says
to remove the rule afterwards.

**6.3** `getTimesAttr().getInt()` returns `int64_t`, which is signed:

```cpp
LogicalResult StrikeOp::verify() {
  if (getTimesAttr().getInt() <= 0)
    return emitOpError("times must be positive, got ") << getTimesAttr().getInt();
  return success();
}
```

Tested: `-1` → `times must be positive, got -1`; `0` → `... got 0`; `3` → ok.

**6.4** (a) `ConfinedAttr<I32Attr, [IntMinValue<1>, IntMaxValue<10>]>`. (b) The `Terminator`
trait. (c) `hasVerifier`: it compares the number of results with an attribute value, which no
ready-made trait does.

---

## Chapter 7

**7.1** With your current `ninja.constant`, **nothing changes**: `%1 = ninja.add %0, %arg0 : f64`
stays. `Commutative` only moves ops that MLIR recognises as constants, and `ninja.constant` is
not `ConstantLike`. After adding `ConstantLike` and a folder to it:

```tablegen
def Ninja_ConstantOp : Ninja_Op<"constant", [Pure, ConstantLike]> {
  ...
  let hasFolder = 1;
}
```
```cpp
OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValueAttr(); }
```

the canonicalizer produces `%1 = ninja.add %arg0, %0 : f64`: the constant moved right.

**7.2**

```tablegen
def Ninja_NegOp : Ninja_Op<"neg", [Pure, Involution, SameOperandsAndResultType]> {
  let arguments = (ins AnyFloat:$input);
  let results = (outs AnyFloat:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}
```

With `Involution`, `neg(neg(x))` is replaced by `x` and both negs disappear. Without it, both
`ninja.neg` ops stay after `--canonicalize`.

**7.3** (a) `SameOperandsElementType` (or `AllElementTypesMatch<["lhs", "rhs"]>`).
(b) `[Terminator, HasParent<"DojoOp">]`.
(c) `TypesMatchWith<"result is the element type of input", "input", "result",
"::llvm::cast<::mlir::ShapedType>($_self).getElementType()">`.

**7.4** `AllTypesMatch<["a", "b", "result"]>` says all three have one type, so once the parser
reads `: f32` it knows the types of `a`, `b` and the result.

**7.5**

```tablegen
def Ninja_PairsOp : Ninja_Op<"pairs", [AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<F32>:$left, Variadic<F32>:$right);
  let assemblyFormat = "`(` $left `)` `(` $right `)` attr-dict";
}
```

```mlir
ninja.pairs (%a, %b) (%c)
// generic:
"ninja.pairs"(%arg2, %arg3, %arg4) <{operandSegmentSizes = array<i32: 2, 1>}> : (f32, f32, f32) -> ()
```

**7.6** Check the **total** number of operands. With `SameVariadicOperandSize` each group's size
is computed as total / 2, so the groups always look equal:

```cpp
LogicalResult ZipListsOp::verify() {
  if ((*this)->getNumOperands() % 2 != 0)
    return emitOpError("expects the same number of xs and ys, got ")
           << (*this)->getNumOperands() << " operands in total";
  return success();
}
```

```
error: 'ninja.zip_lists' op expects the same number of xs and ys, got 3 operands in total
```

---

## Chapter 8

**8.1** With `Pure`, `--cse` merges the two strikes into one. With `MemoryEffects<[MemWrite]>`,
both are kept: writing twice is not the same as writing once.

**8.2** Float division can be `Pure`: dividing by zero gives infinity or NaN, it does not crash.
Integer division by zero is undefined behaviour, so it must not be run "early"; use
`NoMemoryEffect` without `AlwaysSpeculatable`, or implement `ConditionallySpeculatable` to allow
it when the divisor is known not to be zero. MLIR's own `arith` does exactly this: every arith op
gets `NoMemoryEffect` from its base class, `arith.divf` adds `Pure`, and `arith.divsi` /
`arith.divui` use `ConditionallySpeculatable`.

**8.3**

```tablegen
def Ninja_FirstOp : Ninja_Op<"first", [Pure, InferTypeOpAdaptor]> {
  let arguments = (ins AnyType:$a, AnyType:$b);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$a `,` $b attr-dict `:` type($a) `,` type($b)";
}
```

```cpp
LogicalResult FirstOp::inferReturnTypes(MLIRContext *, std::optional<Location>,
                                        FirstOp::Adaptor adaptor,
                                        SmallVectorImpl<Type> &inferred) {
  inferred.push_back(adaptor.getA().getType());
  return success();
}
```

`ninja.first %i, %x : i32, f64` gets result type `i32`.

**8.4**

```tablegen
def Ninja_AddOp : Ninja_Op<"add", [Pure, SameOperandsAndResultType,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> { ... }
```

Add `include "mlir/IR/OpAsmInterface.td"` to the `.td`.

```cpp
void AddOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "total");
}
```

Prints `%total = ninja.add %arg0, %arg0 : f64`.

---

## Chapter 9

**9.1** The four "separate parameter" ones are:

```cpp
static ConstantOp create(OpBuilder &, Location, Type result, FloatAttr value);
static ConstantOp create(OpBuilder &, Location, FloatAttr value);
static ConstantOp create(OpBuilder &, Location, Type result, APFloat value);
static ConstantOp create(OpBuilder &, Location, APFloat value);
```

The easiest today: `ConstantOp::create(builder, loc, builder.getF64FloatAttr(2.5));`.

**9.2** See chapter 9, section 9.4. The new line in `NinjaOps.h.inc` is
`static ConstantOp create(::mlir::OpBuilder &builder, ::mlir::Location location, double v);`.

**9.3**

```tablegen
let builders = [
  OpBuilder<(ins "::mlir::Value":$target), [{
    build($_builder, $_state, target, /*times=*/1u);
  }]>
];
```

`StrikeOp::create(b, loc, x)` now works. Tested: it does not clash with the generated
`ValueRange` builder, because an exact `Value` match is preferred over the conversion.

---

## Chapter 10

**10.1** See chapter 10, section 10.2. `1.0 + 2.0` becomes `ninja.constant 3.000000e+00`.

**10.2** See chapter 10, sections 10.3 and 10.5.

**10.3** No. `AddZero` only checks the right-hand operand, so `0 + x` stays (tested). With
`Commutative`, the canonicalizer first moves the constant to the right, and then `AddZero`
matches.

**10.4** No. `fold` may only return an attribute or an **existing** value; it must not create ops.
`add x, x` is a new op, so this rewrite belongs in a canonicalization pattern.

---

## Chapter 11

**11.1** See chapter 11, section 11.3.

**11.2** `error: 'ninja.yield' op expects parent op 'ninja.dojo'` (from `HasParent<"DojoOp">`).

**11.3** With `[NoTerminator, SingleBlock]` there is no yield at all, and the region cannot
return values. Choose an implicit terminator when the region may one day pass values out (like
`scf.for`'s `scf.yield`); choose `NoTerminator` for a plain container that never returns values
(like `builtin.module`).

**11.4** An empty `else { }` is accepted: `MaxSizedRegion<1>` allows zero blocks.

---

## Chapter 12

**12.1** `if (op.getStance() == Stance::attack) { ... }`.

**12.2** See chapter 12, section 12.2.

**12.3**

```tablegen
def Ninja_ShurikenType : Ninja_Type<"Shuriken", "shuriken"> {
  ...
  let genVerifyDecl = 1;
}
```

```cpp
LogicalResult ShurikenType::verify(function_ref<InFlightDiagnostic()> emitError,
                                   unsigned points) {
  if (points == 0)
    return emitError() << "a shuriken needs at least one point";
  return success();
}
```

```
error: a shuriken needs at least one point
```

---

## Chapter 13

**13.1** See chapter 13.

**13.2** The rule then also matches `f32` adds and builds a `ninja.strike` on an `f32`, which is
invalid. The verifier catches it after the pass, at the original line:

```
error: 'ninja.strike' op operand #0 must be 64-bit float, but got 'f32'
note: see current operation: %0 = "ninja.strike"(%arg0) <{times = 2 : i32}> : (f32) -> f32
```

**13.3**

```tablegen
def StrikeOnce : Pat<(Ninja_StrikeOp $x, ConstantAttr<I32Attr, "1">),
                     (replaceWithValue $x)>;
```

```cpp
void StrikeOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<StrikeOnce>(context);
}
```

with `let hasCanonicalizer = 1;` (instead of `hasCanonicalizeMethod`). Tested: `times 1` becomes
`%x`; `times 2` is left alone.

---

## Chapter 14

**14.1** Your roundtrip test has one function with one strike, so it prints `1 strikes`.

**14.2**

```tablegen
Option<"min", "min", "unsigned", /*default=*/"0",
       "only report functions with at least this many strikes">
```

```cpp
if (static_cast<unsigned>(n) < min)
  return;
```

Several options are separated by spaces: `--ninja-count-strikes=verbose=true min=1`. Tested: only
the functions with a strike are printed.

---

## Chapter 15

**15.1**

```tablegen
class Ninja_BinaryOp<string mnemonic, list<Trait> traits = []>
    : Ninja_Op<mnemonic, !listconcat([Pure, SameOperandsAndResultType], traits)> {
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs);
  let results = (outs AnyFloat:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Ninja_AddOp : Ninja_BinaryOp<"add", [Commutative]> { let summary = "add two floats"; }
def Ninja_SubOp : Ninja_BinaryOp<"sub"> { let summary = "subtract two floats"; }
def Ninja_MulOp : Ninja_BinaryOp<"mul", [Commutative]> { let summary = "multiply two floats"; }
```

Tested: all three parse and print, and your existing tests still pass.

**15.2**

```tablegen
foreach c = ["white", "yellow", "black"] in
  def Belt_ # c { string color = c; }
```

**15.3** Compute `heavy` from the **field**: `bit heavy = !gt(damage, 10);`. Then
`let damage = 99 in def Bomb : Weapon<"bomb", 0>;` gives `heavy = 1` (tested).

---

## Chapter 16

**16.1** The adaptor has `getTarget()`, `getTimes()`, `getTimesAttr()`, `getOperands()` and
`getAttributes()`. `fold` uses a **generic** adaptor whose operand getters return the operands'
**constant values** (attributes) instead of `Value`s, so `fold` can do arithmetic on them.

**16.2** Open-ended.

**16.3**

```
fatal error: ninja/NinjaDialect.h.inc: No such file or directory
```

Without `add_dependencies`, CMake does not know `NinjaDialect.cpp` needs the generated files, so it
may compile it before `mlir-tblgen` has run.

---

## Self-test

1. `(ins AnyType:$a, AnyType:$b)`, `(outs AnyType:$result)`,
   `"$a `,` $b attr-dict `:` functional-type(operands, results)"`.
2. `I32` is a type constraint: the value is an operand (a wire, runtime). `I32Attr` is an
   attribute: a fixed value on the op (compile time).
3. `SameTypeOperands`, `AllTypesMatch<[...]>`, a `PredOpTrait`, or a verifier. `AllTypesMatch`
   (and `SameOperandsAndResultType`) let the format print the type once.
4. `getTimes()` returns `uint32_t`, so `-1` is 4294967295. Fixes:
   `ConfinedAttr<I32Attr, [IntPositive]>`, or compare `getTimesAttr().getInt()`, or use
   `SI32Attr`.
5. `NoMemoryEffect` + `AlwaysSpeculatable`. Integer division: no memory effect, but not safe to
   run early.
6. `ConstantLike` + `hasFolder` on the constant op, `hasFolder` on the op being folded,
   `hasConstantMaterializer` on the dialect.
7. `AttrSizedOperandSegments`. `SameVariadicOperandSize` does not check.
8. Find the `.td` that defines `Foo` (`grep -rln "^def Foo"` in MLIR's include directory) and
   `include` it, plus the matching C++ header.
9. Use `qualified(type($star))` in the format.
10. ODS generates them into `build/include/ninja/NinjaOps.h.inc` / `.cpp.inc`; read them there.
