# 7. Traits

## In one minute

**What.** A trait is a **reusable property you attach to an op** by name, in the list after the op
name: `[Pure, Commutative, SameOperandsAndResultType]`. MLIR provides about 55 of them.

**Why.** The same rules and properties apply to thousands of ops across all dialects: "all
operands have the same type", "the order of the operands does not matter", "this op ends a
block". Writing each rule again for every op would be wasteful, and passes would need to know
every op in every dialect. Instead, the rule is written **once** in MLIR, every op just names it,
and a pass asks "does this op have trait X?" without knowing anything else about the op.

**How.** Each trait is a C++ class that ODS mixes into your op's class. A trait can bring:
- a **check**, which runs during verification (`SameTypeOperands`)
- a **simplification**, which runs during folding (`Involution`: `neg(neg(x))` becomes `x`)
- a **label** that some passes look for (`Vectorizable`)
- extra **C++ methods** on your op

**Where it fits.** In verification, in simplification passes, and in any pass that asks about
traits:
`file.mlir → PARSE → [VERIFY] → [PASSES] → PRINT → output`

**After this chapter** you know what every trait does, can pick the right ones for an op, and know
the traps (traits that need a particular order, traits that do not really check).

Every example in this chapter was built and run. The lab ops are in
[lab/traits/include/ninja/TraitsLab.td](lab/traits/include/ninja/TraitsLab.td).

---

## 7.1 What a trait is

**The idea.** A trait is a **promise** about an op, written in its trait list:

```tablegen
def Ninja_AddOp : Ninja_Op<"add", [Pure, SameOperandsAndResultType]> { ... }
//                                 └──────── the trait list ───────┘
```

**Everyday picture.** Labels on a food package: "vegetarian", "keep frozen". You stick the label
on once, and everyone who handles the package knows how to treat it.

**Traits do one or more of three jobs:**

| Job | Meaning | Example |
|---|---|---|
| **Check** | MLIR verifies the promise and rejects IR that breaks it | `SameTypeOperands` |
| **Optimize** | Passes use the promise to simplify code | `Commutative` |
| **Label** | Nothing is checked; specific passes read the label | `Vectorizable` |

**Where traits come from.** Each trait is a record in some `.td` file. You must `include` that
file (and the matching C++ header in `NinjaDialect.h`), or TableGen says
`Variable not defined`. Most traits in this chapter live in `mlir/IR/OpBase.td`, which you already
include. The exceptions are listed where they appear.

**Order sometimes matters.** Some traits require another trait to come **earlier** in the list.
For example `Scalarizable` without `Elementwise` before it fails with
`Scalarizable requires Elementwise to precede it in traits list`.

---

## 7.2 Traits ODS adds for you

You never write these. ODS adds them from the shape of the op, and you can see them in the
`class ... : public ::mlir::Op<...>` line of `NinjaOps.h.inc`.

Real line for `ninja.strike`:

```cpp
class StrikeOp : public ::mlir::Op<StrikeOp, ::mlir::OpTrait::ZeroRegions,
    ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::FloatType>::Impl,
    ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants,
    ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait,
    ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait,
    ::mlir::InferTypeOpInterface::Trait>
```

| Automatic trait | Added when |
|---|---|
| `ZeroOperands`, `OneOperand`, `NOperands<n>`, `AtLeastNOperands<n>`, `VariadicOperands` | from `arguments` |
| `ZeroResults`, `OneResult`, `NResults<n>`, `VariadicResults`, `OneTypedResult<T>` | from `results` |
| `ZeroRegions`, `OneRegion`, `NRegions<n>`, `VariadicRegions` | from `regions` |
| `ZeroSuccessors`, `OneSuccessor`, ... | from `successors` |
| `OpInvariants` | always: runs the ODS-generated checks |
| `BytecodeOpInterface::Trait` | the op has properties (attributes) |
| `InferTypeOpInterface::Trait` | ODS can work out every result type by itself (here, `F64`) |
| `TokenProducerTrait` / `TokenConsumerTrait` | a result / operand uses the `Token` constraint (7.10) |
| `SingleBlock` | you used `SingleBlockImplicitTerminator` (7.8) |

`Pure` itself is not in the list: it is a **bundle** that expands to
`ConditionallySpeculatable`, `AlwaysSpeculatableImplTrait` and `MemoryEffectOpInterface`
(chapter 8).

---

## 7.3 Algebra traits

These describe mathematical properties. Passes use them to simplify code. Try them with
`--canonicalize`.

### `Commutative`

**Means:** the order of the operands does not matter: `a op b == b op a`.

**Effect:** the canonicalizer moves constants to the **right**, so equivalent code looks the same
and later patterns only have to look for one shape.

"Constant" here means an op MLIR **recognises** as a constant: one with `ConstantLike` and a
folder (chapter 10), like `arith.constant`. Your current `ninja.constant` has neither, so
`Commutative` leaves `ninja.add %k, %x` unchanged when `%k` is a `ninja.constant` (tested). Add
`ConstantLike` and `hasFolder` to it, and the constant moves.

```tablegen
def Ninja_MulOp : Ninja_Op<"mul", [Pure, Commutative, SameOperandsAndResultType]> { ... }
```

```mlir
// before --canonicalize
%k = arith.constant 5.0 : f64
%y = ninja.mul %k, %x : f64

// after
%y = ninja.mul %x, %k : f64
```

### `Idempotent`

**Means:** doing it again changes nothing. For a one-operand op: `f(f(x)) == f(x)`. For a
two-operand op: `f(x, x) == x`.

```tablegen
def Ninja_AbsOp : Ninja_Op<"abs", [Pure, Idempotent, SameOperandsAndResultType]> { ... }
def Ninja_MaxOp : Ninja_Op<"max", [Pure, Idempotent, SameOperandsAndResultType]> { ... }
```

```mlir
// before --canonicalize
%a = ninja.abs %x : f64
%b = ninja.abs %a : f64          // abs(abs(x))
%m = ninja.max %x, %x : f64      // max(x, x)

// after
%a = ninja.abs %x : f64          // %b is now just %a
                                 // %m is now just %x
```

### `Involution`

**Means:** doing it twice gives back the original: `f(f(x)) == x`. Example: negation.

```tablegen
def Ninja_NegOp : Ninja_Op<"neg", [Pure, Involution, SameOperandsAndResultType]> { ... }
```

```mlir
// before --canonicalize
%a = ninja.neg %x : f64
%b = ninja.neg %a : f64          // -(-x)

// after: %b is just %x, and both negs are gone
```

**Watch out:** for both `Idempotent` and `Involution`, MLIR's source says they "should depend on
`SameOperandsAndResultType`". Add that trait too, as in the examples.

### `ConstantLike`

**Means:** "I am a constant: no operands, one result, always the same value."

**Effect:** folding (chapter 10) treats the op as a known value; it is required for constant
folding to work with your dialect.

**Check:** at **C++ compile time**, the op must have one result and zero operands. Putting it on
an op with an operand fails to build with:

```
static assertion failed: expected operation to take zero operands
```

---

## 7.4 Type relations for the whole op

Each of these compares **all** operands, or all operands and results. No names are needed.

### `SameTypeOperands`

**Means:** all operands have exactly the same type.

```mlir
"ninja.pair"(%a, %b) : (f32, f32) -> i1      // ok
"ninja.pair"(%a, %b) : (f32, i32) -> i1
// error: 'ninja.pair' op requires all operands to have the same type
```

### `SameOperandsAndResultType`

**Means:** all operands **and** results have exactly the same type. Lives in
`mlir/Interfaces/InferTypeOpInterface.td` (C++ header `mlir/Interfaces/InferTypeOpInterface.h`).

**Bonus:** the assembly format only needs one type (`ninja.add %a, %b : f32`), and ODS generates
builders that take no result type.

```mlir
"ninja.add"(%a, %x) : (f32, f64) -> f32
// error: 'ninja.add' op requires the same type for all operands and results
```

### `SameOperandsShape`

**Means:** all operands have the same **shape**. Element types may differ.

```mlir
"ninja.zip"(%a, %b) : (tensor<2x3xf32>, tensor<2x3xi8>) -> tensor<9xf32>   // ok
"ninja.zip"(%a, %b) : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<9xf32>
// error: 'ninja.zip' op requires the same shape for all operands
```

### `SameOperandsAndResultShape`

**Means:** all operands and results have the same shape.

```mlir
"ninja.convert"(%a) : (tensor<2x3xf32>) -> tensor<2x3xi32>     // ok
"ninja.convert"(%a) : (tensor<2x3xf32>) -> tensor<3x2xi32>
// error: 'ninja.convert' op requires the same shape for all operands and results
```

### `SameOperandsElementType`

**Means:** all operands have the same **element type**. Shapes may differ.

```mlir
"ninja.concat"(%a, %b) : (tensor<2xf32>, tensor<5xf32>) -> tensor<7xf32>   // ok
"ninja.concat"(%a, %b) : (tensor<2xf32>, tensor<5xi32>) -> tensor<7xf32>
// error: 'ninja.concat' op requires the same element type for all operands
```

### `SameOperandsAndResultElementType`

**Means:** all operands and results have the same element type.

```mlir
"ninja.reshape"(%a) : (tensor<6xf32>) -> tensor<2x3xf32>     // ok
"ninja.reshape"(%a) : (tensor<6xf32>) -> tensor<2x3xi32>
// error: 'ninja.reshape' op requires the same element type for all operands and results
```

### `SameOperandsAndResultRank`

**Means:** all operands and results have the same **rank** (number of dimensions). Lives in
`InferTypeOpInterface.td`.

```mlir
"ninja.transpose"(%a) : (tensor<2x3xf32>) -> tensor<3x2xf32>   // ok: both rank 2
"ninja.transpose"(%a) : (tensor<2x3xf32>) -> tensor<6xf32>
// error: 'ninja.transpose' op result type has different rank than operands
```

### `ResultsAreFloatLike`

**Means:** every result is a float, or a vector/tensor of floats.

```mlir
"ninja.to_float"(%a) : (i32) -> f32               // ok
"ninja.to_float"(%a) : (i32) -> tensor<4xf16>     // ok
"ninja.to_float"(%a) : (f32) -> i32
// error: 'ninja.to_float' op requires a floating point type
```

### `ResultsBroadcastableShape`

**Means:** the operands' shapes are broadcast-compatible (NumPy rules), and the result shape
matches the broadcast shape. **C++ header:** `mlir/Dialect/Traits.h`.

```mlir
"ninja.badd"(%a, %b) : (tensor<2x1xf32>, tensor<3xf32>) -> tensor<2x3xf32>   // ok
"ninja.badd"(%a, %b) : (tensor<2x4xf32>, tensor<3xf32>) -> tensor<2x3xf32>
// error: 'ninja.badd' op operands don't have broadcast-compatible shapes
"ninja.badd"(%a, %b) : (tensor<2x1xf32>, tensor<3xf32>) -> tensor<5x3xf32>
// error: 'ninja.badd' op result type '5x3' not broadcast compatible with broadcasted operands's shapes '2x3'
```

### `FirstAttrDerivedResultType`

**Means:** the result type comes from the **first attribute**: its type, or, if it is a
`TypeAttr`, the type it holds.

**Effect:** no runtime check. ODS adds extra **builders** that take no result type and compute it
from the attribute:

```cpp
// generated for ninja.typed_const, which has (ins TypedAttrInterface:$value)
static void build(OpBuilder &, OperationState &, ValueRange operands,
                  ArrayRef<NamedAttribute> attributes);  // no result type needed
```

---

## 7.5 Type relations between named values

These name the operands/results they compare, so they work on any subset. The names are the
`$names` from `arguments` and `results`.

### `AllTypesMatch<["a", "b", ...]>`

**Means:** the named values all have exactly the same type.

**Bonus:** the assembly format can print the type once.

```tablegen
def Ninja_SelectOp : Ninja_Op<"select", [AllTypesMatch<["a", "b", "result"]>]> {
  let arguments = (ins I1:$cond, AnyType:$a, AnyType:$b);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$cond `,` $a `,` $b attr-dict `:` type($result)";
}
```

```mlir
%r = ninja.select %c, %a, %b : f32                              // ok
"ninja.select"(%c, %a, %b) : (i1, f32, i32) -> f32
// error: 'ninja.select' op failed to verify that all of {a, b, result} have same type
```

### `AllElementTypesMatch<[...]>`

**Means:** the named values have the same element type.

```mlir
"ninja.stack"(%a, %b) : (tensor<2xf32>, tensor<3x3xf32>) -> tensor<9xi8>    // ok
"ninja.stack"(%a, %b) : (tensor<2xf32>, tensor<2xf64>) -> tensor<9xi8>
// error: 'ninja.stack' op failed to verify that all of {a, b} have same element type
```

### `AllShapesMatch<[...]>`

**Means:** the named values have the same shape.

```mlir
"ninja.bitcast"(%a) : (tensor<2x3xf32>) -> tensor<2x3xi32>      // ok
"ninja.bitcast"(%a) : (tensor<2x3xf32>) -> tensor<6xi32>
// error: 'ninja.bitcast' op failed to verify that all of {input, result} have same shape
```

### `AllRanksMatch<[...]>`

**Means:** the named values have the same rank.

```mlir
"ninja.permute"(%a) : (tensor<2x3xf32>) -> tensor<3x2xi8>       // ok
"ninja.permute"(%a) : (tensor<2x3xf32>) -> tensor<6xf32>
// error: 'ninja.permute' op failed to verify that all of {input, result} have same rank
```

### `AllElementCountsMatch<[...]>`

**Means:** the named values hold the same **number of elements** (2x3 = 6).

```mlir
"ninja.flatten"(%a) : (tensor<2x3xf32>) -> tensor<6xf32>        // ok
"ninja.flatten"(%a) : (tensor<2x3xf32>) -> tensor<5xf32>
// error: 'ninja.flatten' op failed to verify that all of {input, result} have same element count
```

### `AllTypeIDsMatch<[...]>`

**Means:** the named values have the same **kind** of type. `i8` and `i64` are both integer
types, so they match; `i8` and `f32` do not.

```mlir
"ninja.same_kind"(%a, %b) : (i8, i64) -> ()                     // ok
"ninja.same_kind"(%a, %b) : (i8, f32) -> ()
// error: 'ninja.same_kind' op failed to verify that all of {a, b} have same type IDs
```

### `TypesMatchWith<"summary", "from", "to", "transform">`

**Means:** `transform(type of from) == type of to`. In `transform`, `$_self` stands for the type
of `from`.

**Bonus:** the assembly format can leave out the `to` type.

```tablegen
def Ninja_ExtractOp : Ninja_Op<"extract", [
    TypesMatchWith<"result is the element type of input", "input", "result",
                   "::llvm::cast<::mlir::ShapedType>($_self).getElementType()">]> {
  let arguments = (ins AnyRankedTensor:$input, Index:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `[` $index `]` attr-dict `:` type($input)";
}
```

```mlir
%r = ninja.extract %t[%i] : tensor<4xf32>          // result type f32 is worked out
"ninja.extract"(%t, %i) : (tensor<4xf32>, index) -> i32
// error: 'ninja.extract' op failed to verify that result is the element type of input
```

### `OptionalTypesMatchWith<...>`

**Means:** the same as `TypesMatchWith`, but passes when either value is a missing `Optional`.

```mlir
"ninja.extract_or"(%t) : (tensor<4xf32>) -> f32               // no fallback: ok
"ninja.extract_or"(%t, %d) : (tensor<4xf32>, f32) -> f32      // ok
"ninja.extract_or"(%t, %d) : (tensor<4xf32>, i32) -> f32
// error: 'ninja.extract_or' op failed to verify that fallback is the element type of input
```

### `RangedTypesMatchWith<...>`

**Means:** `TypesMatchWith` for **lists** (`Variadic`), compared element by element.

```tablegen
RangedTypesMatchWith<"result types match input types", "inputs", "results", "$_self">
```

```mlir
%r:2 = ninja.copy_all %a, %b : f32, i32                 // results are (f32, i32)
"ninja.copy_all"(%a, %b) : (f32, i32) -> (f32, f32)
// error: 'ninja.copy_all' op failed to verify that result types match input types
```

### `TypeIs<"name", Type>`

**Means:** the named value has exactly this type.

```tablegen
TypeIs<"result", I32>
```

```mlir
"ninja.count"(%t) : (tensor<4xf32>) -> i64
// error: 'ninja.count' op failed to verify that 'result' is 32-bit signless integer
```

(You could also write `I32:$result` directly. `TypeIs` is useful when the constraint on the value
itself must stay loose, for example for an interface or a shared base class.)

### `ElementTypeIs<"name", Type>`

**Means:** the named value's element type is exactly this type.

```mlir
"ninja.relu"(%t) : (tensor<4xf16>) -> tensor<4xf16>
// error: 'ninja.relu' op failed to verify that 'input' is 32-bit float
```

### `ShapedTypeMatchesElementCountAndTypes<"shaped", "elements">`

**Means:** the tensor `shaped` has as many elements as the list `elements`, and its element type
matches theirs. Used by ops that build a tensor from scalars.

```mlir
"ninja.from_elements"(%a, %b, %c) : (f32, f32, f32) -> tensor<3xf32>   // ok
"ninja.from_elements"(%a, %b) : (f32, f32) -> tensor<3xf32>
// error: 'ninja.from_elements' op failed to verify that shaped type 'result' matches 'elements' element count and types
```

### `InputMatchesTypes<["a", "b"], [TypeA, TypeB]>`

**Means:** each named operand matches its type, position by position.

```tablegen
InputMatchesTypes<["value", "factor"], [F32, I32]>
```

```mlir
"ninja.scale_by"(%a, %b) : (f32, i32) -> ()      // ok
"ninja.scale_by"(%a, %b) : (f32, f32) -> ()
// error: 'ninja.scale_by' op failed to verify that operands {value, factor} match expected types
```

### `InputAddressIsCombinationOf<["a", "b"], [[...], [...]]>`

**Means:** the named operands match **one of** the listed type combinations.

```tablegen
InputAddressIsCombinationOf<["a", "b"], [[F32, F32], [F64, I64]]>
```

```mlir
"ninja.combo"(%a, %b) : (f64, i64) -> ()         // ok: second combination
"ninja.combo"(%a, %b) : (f64, f64) -> ()
// error: 'ninja.combo' op failed to verify that operands {a, b} match one of the allowed type combinations
```

---

## 7.6 Building your own check traits

When no ready-made trait fits, build one from a **predicate**: a piece of C++ that returns `true`
or `false`. Inside it, `$name` means "the value called name".

### `PredOpTrait<"description", predicate>`

```tablegen
def Ninja_MatmulOp : Ninja_Op<"matmul", [
    PredOpTrait<"lhs is a rank-2 tensor",
                CPred<"::llvm::cast<::mlir::ShapedType>($lhs.getType()).getRank() == 2">>]> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
}
```

```mlir
"ninja.matmul"(%a, %b) : (tensor<6xf32>, tensor<3xf32>) -> tensor<2xf32>
// error: 'ninja.matmul' op failed to verify that lhs is a rank-2 tensor
```

`CPred<"...">` is a predicate written in C++. You can combine predicates with `And<[...]>`,
`Or<[...]>` and `Neg<...>`.

### `AllMatch<["expr1", "expr2", ...], "description">`

**Means:** all the C++ expressions give equal results.

```tablegen
AllMatch<["$a.getType()", "$b.getType()"], "a and b have the same type">
```

```mlir
"ninja.all_match"(%a, %b) : (f32, f16) -> ()
// error: 'ninja.all_match' op failed to verify that a and b have the same type
```

### `AllMatchSameOperatorTrait<[names], "operator", "summary">`

**Means:** apply the same C++ expression to each named value; all results must be equal. In
`operator`, `$_self` stands for each value.

```tablegen
AllMatchSameOperatorTrait<["a", "b"], "$_self.getType().getIntOrFloatBitWidth()", "bit width">
```

```mlir
"ninja.same_width"(%a, %b) : (i32, i8) -> ()
// error: 'ninja.same_width' op failed to verify that all of {a, b} have same bit width
```

`AllTypesMatch`, `AllShapesMatch` and the others in 7.5 are built from this one.

### `AnyMatchOperatorTrait<[names], "operator", "summary">`

**Means:** the C++ expression must be true for **at least one** named value.

```tablegen
AnyMatchOperatorTrait<["a", "b"], "::llvm::isa<::mlir::FloatType>($_self.getType())", "float type">
```

```mlir
"ninja.any_float"(%a, %b) : (i32, f32) -> ()     // ok: b is a float
"ninja.any_float"(%a, %b) : (i32, i8) -> ()
// error: 'ninja.any_float' op failed to verify that any of {a, b} has float type
```

**Watch out:** the generated checks call helpers like `getElementTypeOrSelf`. Include
`mlir/IR/TypeUtilities.h` in `NinjaDialect.h`, or the build fails with
`'getElementTypeOrSelf' was not declared in this scope`.

---

## 7.7 The elementwise family

For ops that work **one element at a time**, like `x * 2` applied to every element of a tensor.

### `Elementwise`

**Means:** the op applies to each element independently.

**Checks** (all real messages):

```mlir
"ninja.elt"(%a) : (f32) -> f32                                  // ok: scalars
"ninja.elt"(%a) : (tensor<4xf32>) -> tensor<4xf32>              // ok
"ninja.elt"(%a) : (vector<4xf32>) -> vector<4xf32>              // ok

"ninja.elt"(%a) : (tensor<4xf32>) -> f32
// error: 'ninja.elt' op if an operand is non-scalar, then there must be at least one non-scalar result
"ninja.elt"(%a) : (f32) -> tensor<4xf32>
// error: 'ninja.elt' op if a result is non-scalar, then at least one operand must be non-scalar
"ninja.elt"(%a) : (tensor<4xf32>) -> vector<4xf32>
// error: 'ninja.elt' op all non-scalar operands/results must have the same shape and base type
"ninja.elt"(%a) : (tensor<4xf32>) -> tensor<5xf32>
// error: 'ninja.elt' op all non-scalar operands/results must have the same shape and base type
```

No broadcasting: all tensors/vectors must have the same shape.

### `Scalarizable`, `Vectorizable`, `Tensorizable`

**Labels** (no extra checks). They promise a transformation is safe:

| Trait | Promise |
|---|---|
| `Scalarizable` | the tensor/vector version can be rewritten to work on single scalars |
| `Vectorizable` | the scalar version can be rewritten to work on vectors |
| `Tensorizable` | the scalar version can be rewritten to work on tensors |

**Rule:** each requires `Elementwise` **before** it in the list:

```tablegen
[Elementwise, Scalarizable]       // ok
[Scalarizable]
// error: Scalarizable requires Elementwise to precede it in traits list
```

### `ElementwiseMappable`

**Means:** a bundle of all four: `Elementwise, Scalarizable, Vectorizable, Tensorizable`. This is
what MLIR's own `arith` ops use.

```tablegen
def Ninja_SigmoidOp : Ninja_Op<"sigmoid", [Pure, ElementwiseMappable]> { ... }
```

The checks are those of `Elementwise`.

---

## 7.8 Regions and nesting

How ops with regions behave, and where ops may be placed. Chapter 11 covers regions in depth.

### `Terminator`

**Means:** this op must be the **last** op in its block, like `func.return`.

```mlir
ninja.dojo {
  ninja.yield
  ninja.bow
}
// error: 'ninja.yield' op must be the last operation in the parent block
```

### `NoTerminator`

**Means:** this op's regions do **not** need a terminator at the end. Requires one block
(use with `SingleBlock`). `builtin.module` uses it.

```tablegen
def Ninja_ScrollOp : Ninja_Op<"scroll", [NoTerminator, SingleBlock]> { ... }
```

```mlir
ninja.scroll {
  %y = ninja.neg %x : f64          // no terminator needed
}
```

### `SingleBlock`

**Means:** each region has at most **one** block.

```mlir
"ninja.one_block"() ({
^a:
  ...
^b:
}) : () -> ()
// error: 'ninja.one_block' op expects region #0 to have 0 or 1 blocks
```

### `SingleBlockImplicitTerminator<"YieldOp">`

**Means:** one block, always ending with the named terminator, **which you may leave out** when
writing text: MLIR adds it for you. It includes `SingleBlock`.

```mlir
ninja.dojo {
  %z = ninja.constant 1.0          // no ninja.yield written...
}
```

```mlir
// ...but it is there in the generic form
"ninja.dojo"() ({
  %3 = "ninja.constant"() <{value = 1.000000e+00 : f64}> : () -> f64
  "ninja.yield"() : () -> ()
}) : () -> ()
```

The terminator op needs a builder with no arguments, which ODS generates automatically when the
op has no operands or attributes.

### `NoRegionArguments`

**Means:** the region's blocks take no arguments (`^bb0(%a: f32)` is not allowed).

```mlir
"ninja.no_args"() ({
^bb0(%a: f32):
}) : () -> ()
// error: 'ninja.no_args' op region should have no arguments
```

### `IsolatedFromAbove`

**Means:** code inside the region **cannot use values defined outside** it. Functions have this:
a function body cannot use another function's values.

**Why it matters:** MLIR can run passes on isolated ops in parallel, and a pass manager can be
nested on them.

```mlir
func.func @f(%x: f64) {
  ninja.sealed {
    %y = ninja.neg %x : f64        // %x comes from outside
  }
}
// error: 'ninja.neg' op using value defined outside the region
```

A region without this trait, like `ninja.dojo`, may use `%x`.

### `HasParent<"OpName">`

**Means:** the op's **direct** parent must be that op.

```mlir
ninja.dojo { ninja.bow }                        // ok
ninja.dojo { ninja.scroll { ninja.bow } }
// error: 'ninja.bow' op expects parent op 'ninja.dojo'
```

### `ParentOneOf<["OpA", "OpB"]>`

**Means:** the direct parent must be one of these.

```mlir
func.func @f() { ninja.yield ... }
// error: 'ninja.yield' op expects parent op to be one of 'ninja.dojo, ninja.temple'
```

### `HasAncestor<"OpName">`

**Means:** the op must be **somewhere** inside that op, at any depth.

```mlir
ninja.dojo { ninja.scroll { ninja.meditate } }  // ok: dojo is an ancestor
ninja.scroll { ninja.meditate }
// error: 'ninja.meditate' op expects ancestor op 'ninja.dojo'
```

### `AncestorOneOf<["OpA", "OpB"]>`

**Means:** somewhere inside one of these.

```mlir
func.func @f() { ninja.rest ... }
// error: 'ninja.rest' op expects ancestor op to be one of 'ninja.dojo, ninja.temple'
```

**Op names in these traits** are C++ class names (`"DojoOp"`), not IR names (`"ninja.dojo"`).

---

## 7.9 Ops with several variadic groups

With one `Variadic` operand, MLIR knows that everything left over belongs to it. With **two or
more** `Variadic`/`Optional` groups, it needs to know where each group starts. Pick one of these
traits.

### `AttrSizedOperandSegments`

**Means:** a hidden property `operandSegmentSizes` records the size of each group. **This is the
safe default.**

```tablegen
def Ninja_ThrowAtOp : Ninja_Op<"throw_at", [AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<F32>:$targets, Optional<I32>:$times);
  let assemblyFormat = "`(` $targets `)` (`times` $times^)? attr-dict";
}
```

```mlir
ninja.throw_at (%a, %b, %c) times %t
// generic form: the sizes are stored for you
"ninja.throw_at"(%arg0, %arg1, %arg2, %arg3) <{operandSegmentSizes = array<i32: 3, 1>}> : ...

ninja.throw_at (%a, %b)
"ninja.throw_at"(%arg0, %arg1) <{operandSegmentSizes = array<i32: 2, 0>}> : ...
```

Every group gets an entry, even non-variadic ones (always `1`). Wrong sizes are rejected:

```
error: 'ninja.throw_at' op operand count (2) does not match with the total size (1)
       specified in attribute 'operandSegmentSizes'
```

### `AttrSizedResultSegments`

The same for results, with a property called `resultSegmentSizes`.

```mlir
%r:3 = "ninja.unpack"(%n) <{resultSegmentSizes = array<i32: 2, 1>}> : (i32) -> (f32, f32, i32)
```

### `SameVariadicOperandSize`

**Means:** all variadic operand groups have the **same** size, so MLIR splits the operands evenly
and needs no stored sizes.

```tablegen
def Ninja_ZipListsOp : Ninja_Op<"zip_lists", [SameVariadicOperandSize]> {
  let arguments = (ins Variadic<F32>:$xs, Variadic<F32>:$ys);
  let assemblyFormat = "`(` $xs `)` `(` $ys `)` attr-dict";
}
```

```mlir
ninja.zip_lists (%a, %b) (%c, %d)      // xs = a, b; ys = c, d
```

**Watch out: the size is not checked.** It is a promise you must keep. With 3 operands, MLIR
divides 3 by 2, and the third operand is silently lost when printed:

```mlir
ninja.zip_lists (%a, %b) (%c)          // accepted, no error...
ninja.zip_lists(%arg0) (%arg1)         // ...and printed back like this: %c is gone
```

If you use this trait, add a verifier that checks the sizes, or use `AttrSizedOperandSegments`.

### `SameVariadicResultSize`

The same for results, with the same lack of checking.

---

## 7.10 Scopes and markers

### `AutomaticAllocationScope`

**Means:** stack memory allocated inside this op (with `memref.alloca`) is freed when control
leaves the op's region. Functions have this trait.

**Check:** `memref.alloca` requires an ancestor with this trait:

```mlir
ninja.scroll { %m = memref.alloca() : memref<4xf32> }
// error: 'memref.alloca' op requires an ancestor op with AutomaticAllocationScope trait

ninja.scope { %m = memref.alloca() : memref<4xf32> }       // ok: scope has the trait
```

At C++ compile time the op must have at least one region (`expected operation to have one or more
regions`).

### `AffineScope`

**Means:** the op starts a new scope for the `affine` dialect. Inside it, `index` values defined
at its top level (or as its region arguments) count as **symbols**, which affine ops like
`affine.for` and `affine.load` may use in their bounds and indices. `func.func` has this trait.
You only need it if your op holds affine code. Like `AutomaticAllocationScope`, it requires at
least one region.

### `MemRefsNormalizable`

**Label.** The op can be rewritten safely when its `memref` operands have a non-identity layout
map. The `-normalize-memrefs` pass only touches ops with this trait.

### `TokenProducerTrait`, `TokenConsumerTrait`

MLIR has a builtin `token` type. A token carries no data; it is a handle to an op or a region,
used for things like coroutine bookkeeping. Tokens are excluded from `AnyType`.

**Rule:** only ops with `TokenProducerTrait` may produce tokens (as results, or as region entry
block arguments), and only ops with `TokenConsumerTrait` may use them.

**ODS adds these for you** when an operand or result uses the `Token` constraint:

```tablegen
def Ninja_MakeTokenOp : Ninja_Op<"make_token"> {     // no trait written...
  let results = (outs Token:$token);                 // ...but Token adds TokenProducerTrait
}
```

You write them yourself only when the type is not declared as `Token`, for example a result that
accepts any type:

```tablegen
def Ninja_AnythingOp : Ninja_Op<"anything"> {
  let results = (outs Type<CPred<"true">>:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}
```

```mlir
%t = ninja.anything : token
// error: 'ninja.anything' op produces token result #0 but does not have the TokenProducerTrait
```

With `[TokenProducerTrait]` added, the same line is accepted.

---

## 7.11 Symbols and region kinds

### `SymbolTable`

**Means:** the op's region is a **symbol table**: a namespace where ops can be named with `@name`.
Names inside must be unique. `builtin.module` has this.

### `Symbol` (plus `SymbolName` and `SymbolVisibility`)

**Means:** the op **defines** a name in the nearest symbol table, like `func.func @main`.

In this LLVM version, `Symbol` must be used with `SymbolName` and `SymbolVisibility`, and the op
must have `sym_name` and `sym_visibility` attributes. With `Symbol` alone, the C++ fails to
compile (`has no member named 'getNameAttr'`). Include `mlir/IR/SymbolInterfaces.td`; C++ header
`mlir/IR/SymbolTable.h`.

```tablegen
def Ninja_ClanOp : Ninja_Op<"clan", [SymbolTable, NoTerminator, SingleBlock]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def Ninja_MemberOp : Ninja_Op<"member", [Symbol, SymbolName, SymbolVisibility]> {
  let arguments = (ins SymbolNameAttr:$sym_name, OptionalAttr<StrAttr>:$sym_visibility);
  let assemblyFormat = "($sym_visibility^)? $sym_name attr-dict";
}
```

```mlir
ninja.clan {
  ninja.member @kenji
  ninja.member "private" @aiko
}
```

Checks you get:

```
ninja.clan { ninja.member @kenji
             ninja.member @kenji }
error: redefinition of symbol named 'kenji'

ninja.member "secret" @kenji
error: 'ninja.member' op visibility expected to be one of ["public", "private", "nested"], but got "secret"

func.func @f() { ninja.member @lonely ... }
error: 'ninja.member' op symbol's parent must have the SymbolTable trait
```

### Graph regions: `HasOnlyGraphRegion`, `GraphRegionNoTerminator`

Normal regions are **SSACFG** regions: a value must be defined before it is used. A **graph
region** has no order: ops may use values defined later, as in a circuit diagram. Include
`mlir/IR/RegionKindInterface.td`; C++ header `mlir/IR/RegionKindInterface.h`.

```mlir
// in a normal region
ninja.dojo { %a = ninja.neg %b : f64
             %b = ninja.neg %a : f64 }
// error: operand #0 does not dominate this use

// in a graph region: accepted
ninja.graph_noterm { %a = ninja.neg %b : f64
                     %b = ninja.neg %a : f64 }
```

- **`GraphRegionNoTerminator`** is the easy way: a bundle of `NoTerminator`, `SingleBlock`,
  `RegionKindInterface` and `HasOnlyGraphRegion`.
- **`HasOnlyGraphRegion`** alone is **not enough**. It answers "what kind of region?" through
  `RegionKindInterface`, but does not add that interface. Without it, the region is still treated
  as normal and you get the dominance error. Use `[RegionKindInterface, HasOnlyGraphRegion, ...]`.

---

## 7.12 Quick reference

| Trait | Kind | One-line meaning |
|---|---|---|
| `Commutative` | optimize | `a op b == b op a` |
| `Idempotent` | optimize | `f(f(x)) == f(x)` or `f(x, x) == x` |
| `Involution` | optimize | `f(f(x)) == x` |
| `ConstantLike` | optimize | a constant: no operands, one result |
| `SameTypeOperands` | check | all operands same type |
| `SameOperandsAndResultType` | check | all operands and results same type |
| `SameOperandsShape` | check | all operands same shape |
| `SameOperandsAndResultShape` | check | operands and results same shape |
| `SameOperandsElementType` | check | operands same element type |
| `SameOperandsAndResultElementType` | check | operands and results same element type |
| `SameOperandsAndResultRank` | check | operands and results same rank |
| `ResultsAreFloatLike` | check | results are floats |
| `ResultsBroadcastableShape` | check | NumPy-style broadcasting |
| `FirstAttrDerivedResultType` | builders | result type from first attribute |
| `AllTypesMatch` | check | named values same type |
| `AllElementTypesMatch` | check | named values same element type |
| `AllShapesMatch` | check | named values same shape |
| `AllRanksMatch` | check | named values same rank |
| `AllElementCountsMatch` | check | named values same element count |
| `AllTypeIDsMatch` | check | named values same kind of type |
| `TypesMatchWith` | check | `transform(type of a) == type of b` |
| `OptionalTypesMatchWith` | check | same, skipped if missing |
| `RangedTypesMatchWith` | check | same, for lists |
| `TypeIs` | check | named value has this type |
| `ElementTypeIs` | check | named value has this element type |
| `ShapedTypeMatchesElementCountAndTypes` | check | tensor matches list of scalars |
| `InputMatchesTypes` | check | operands match types by position |
| `InputAddressIsCombinationOf` | check | operands match one allowed combination |
| `PredOpTrait` | check | your own C++ condition |
| `AllMatch` | check | C++ expressions all equal |
| `AllMatchSameOperatorTrait` | check | same expression equal for all values |
| `AnyMatchOperatorTrait` | check | expression true for at least one value |
| `Elementwise` | check | per-element op, matching shapes |
| `Scalarizable` / `Vectorizable` / `Tensorizable` | label | safe to rewrite between scalar/vector/tensor |
| `ElementwiseMappable` | bundle | all four above |
| `Terminator` | check | must be last in its block |
| `NoTerminator` | structure | regions need no terminator |
| `SingleBlock` | check | one block per region |
| `SingleBlockImplicitTerminator` | structure | one block, terminator added for you |
| `NoRegionArguments` | check | region blocks take no arguments |
| `IsolatedFromAbove` | check | region cannot use outside values |
| `HasParent` / `ParentOneOf` | check | direct parent must be... |
| `HasAncestor` / `AncestorOneOf` | check | some ancestor must be... |
| `AttrSizedOperandSegments` | structure | stored sizes for operand groups |
| `AttrSizedResultSegments` | structure | stored sizes for result groups |
| `SameVariadicOperandSize` | structure | equal operand groups (**not checked**) |
| `SameVariadicResultSize` | structure | equal result groups (**not checked**) |
| `AutomaticAllocationScope` | scope | `alloca` memory freed on exit |
| `AffineScope` | scope | new scope for affine symbols |
| `MemRefsNormalizable` | label | safe for `-normalize-memrefs` |
| `TokenProducerTrait` / `TokenConsumerTrait` | check | may produce / consume `token` |
| `SymbolTable` | structure | a namespace for `@names` |
| `Symbol` + `SymbolName` + `SymbolVisibility` | structure | defines an `@name` |
| `HasOnlyGraphRegion` / `GraphRegionNoTerminator` | structure | graph regions |

---

## Exercises

**7.1** Add `Commutative` to `ninja.add`. Write a test where a constant is the first operand and
run `--canonicalize`. What changes?

**7.2** Create `ninja.neg` with `Involution`, and check that `neg(neg(x))` disappears under
`--canonicalize`. Then remove `Involution` and check that it no longer does.

**7.3** Choose the trait for each: (a) an op whose two tensor operands must have the same element
type but may have different shapes; (b) a terminator that may only appear inside `ninja.dojo`;
(c) an op whose result type is always the element type of its first operand.

**7.4** Make `ninja.select` from section 7.5. Why does its assembly format need only one type?

**7.5** Write an op with two `Variadic<F32>` operand groups and `AttrSizedOperandSegments`. Print
it with `--mlir-print-op-generic` and find `operandSegmentSizes`.

**7.6** Show the `SameVariadicOperandSize` bug from section 7.9 yourself. Then add a
`hasVerifier` that rejects unequal groups. Careful: comparing `getXs().size()` with
`getYs().size()` does **not** work, because both are computed as half the total (with 3
operands, both report 1). What should you check instead?

[Solutions](solutions.md#chapter-7)
