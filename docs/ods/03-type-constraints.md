# 3. Type constraints

## In one minute

**What.** A type constraint is a rule on an operand or result that says **which types are
allowed**: `F64` (only `f64`), `AnyFloat` (any float), `Variadic<F64>` (any number of `f64`s).

**Why.** An op only makes sense for some types. `ninja.strike` multiplies a float; given an `i32`,
any pass that trusts it would compute nonsense or crash. A type constraint catches the mistake
**at the op that is wrong**, with a clear message, so every later pass can trust the types.

**How.** ODS turns each constraint into a C++ check inside the op's generated
`verifyInvariants()` function, which runs during verification. The constraint also decides the
C++ type of the getter (`F64:$target` gives `TypedValue<FloatType> getTarget()`), and, when it is
one exact type, lets the parser and builders fill the type in by themselves.

**Where it fits.** Mostly in verification, with help to parsing and to C++ code:
`file.mlir → PARSE → [VERIFY] → PASSES → PRINT → output`

**After this chapter** you can choose the right constraint for every operand and result, including
"any number of" and "may be missing".

---

## 3.1 What a type constraint is

**The idea.** Every value in MLIR has a type: `f32`, `f64`, `i32`, `tensor<2x2xf32>`, and so on. A
**type constraint** is a rule on an operand or result that says **which types are allowed**. You
write it once, and ODS writes the C++ check for you.

**Everyday picture.** A coin slot only takes coins. The slot is the operand; "coins only" is the
type constraint.

In C++ terms: `double add(double a, double b)` will not accept a string. `AnyFloat:$lhs` means
the same thing for an MLIR op.

**Syntax.** The constraint goes **before** the colon, and the name **after** it:

```
AnyFloat : $lhs
   │         └── the name. Gives you getLhs() in C++
   └──────────── the constraint. Says which types are allowed
```

---

## 3.2 Example

```tablegen
def Ninja_AddOp : Ninja_Op<"add", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs);   // inputs: any float type
  let results   = (outs AnyFloat:$result);             // output: any float type
}

def Ninja_StrikeOp : Ninja_Op<"strike", [Pure]> {
  let arguments = (ins F64:$target, I32Attr:$times);   // $target: exactly f64
  let results   = (outs F64:$result);
}
```

**What you see.** Accepted:

```mlir
%0 = ninja.add %a, %b : f32                    // f32 is a float
%1 = ninja.add %x, %x : f64                    // so is f64
%2 = ninja.strike %x : f64 times 2 : f64       // exactly f64
```

Rejected, with no C++ written by you:

```mlir
%0 = "ninja.add"(%i, %i) : (i32, i32) -> i32
// error: 'ninja.add' op operand #0 must be floating-point, but got 'i32'

%0 = "ninja.strike"(%a) <{times = 2 : i32}> : (f32) -> f64
// error: 'ninja.strike' op operand #0 must be 64-bit float, but got 'f32'
```

These use the generic `"ninja.add"(...)` form on purpose. With the pretty form,
`ninja.add %i, %i : i32` fails earlier, while the text is being read, with a less helpful
message: `invalid kind of type specified`.

---

## 3.3 A constraint checks one value; a trait checks how values relate

```mlir
%0 = "ninja.add"(%a, %x) : (f32, f64) -> f32
// error: 'ninja.add' op requires the same type for all operands and results
```

Each operand passes `AnyFloat` on its own. The error comes from the **trait**
`SameOperandsAndResultType`, which compares them. Constraints look at one value; traits
(chapter 7) can look at several.

---

## 3.4 The constraints you will use

**Exact types**

| Constraint | Accepts |
|---|---|
| `F16`, `BF16`, `F32`, `F64` | that float type |
| `I1`, `I8`, `I16`, `I32`, `I64` | that signless integer type |
| `SI32`, `UI32` (and other widths) | signed / unsigned integer of that width |
| `Index` | `index` |
| `NoneType` | `none` |
| `Token` | the builtin `token` type (see chapter 7, `TokenProducerTrait`) |

**Families**

| Constraint | Accepts |
|---|---|
| `AnyType` | any type **except** `token` |
| `AnyFloat` | any float type |
| `AnyInteger` | any integer type: signless, signed or unsigned |
| `AnySignlessInteger` | `i1`, `i8`, `i32`, ... (not `si32`/`ui32`) |
| `AnySignlessIntegerOrIndex` | the above, plus `index` |
| `AnyComplex` | `complex<...>` |

**Containers**

| Constraint | Accepts |
|---|---|
| `AnyTensor` | any tensor, ranked or unranked |
| `AnyRankedTensor` | `tensor<2x3xf32>`, `tensor<?xf32>`, ... |
| `AnyUnrankedTensor` | `tensor<*xf32>` |
| `AnyStaticShapeTensor` | a tensor with no `?` in its shape |
| `TensorOf<[F32, F64]>` | a tensor whose elements are `f32` or `f64` |
| `RankedTensorOf<[F32]>` | a ranked tensor of `f32` |
| `TensorRankOf<[F32], [2]>` | a rank-2 tensor of `f32` |
| `AnyMemRef`, `MemRefOf<[F32]>` | memrefs |
| `AnyVectorOfAnyRank`, `VectorOfAnyRankOf<[F32]>` | vectors |
| `AnyShaped` | any tensor, memref or vector |

**Combining**

| Constraint | Accepts |
|---|---|
| `AnyTypeOf<[F32, I32]>` | `f32` **or** `i32` |
| `AllOfType<[...]>` | must pass **all** listed constraints |

**Strict or loose.** Pick the strictest constraint that is still true:

```
AnyType  ⊃  AnyFloat              ⊃  F64
(anything)  (f16, f32, f64, ...)     (only f64)
```

Too loose lets bad IR in, which then crashes a later pass. Too strict rejects valid IR.

---

## 3.5 `Variadic`: any number of operands

**The idea.** `Variadic<X>` means "zero or more operands, each of type `X`".

**Everyday picture.** A shopping bag: it can hold no items, one, or many.

**Example** *(lab op)*:

```tablegen
def Ninja_SumOp : Ninja_Op<"sum", [Pure]> {
  let arguments = (ins Variadic<F64>:$inputs);
  let results = (outs F64:$result);
  let assemblyFormat = "$inputs attr-dict";
}
```

```mlir
%s = ninja.sum %a, %b, %c          // three operands
```

In C++, `getInputs()` returns all of them as a range you can loop over.

`Variadic` also works for results: `(outs Variadic<F32>:$parts)`.

**Watch out.** An op with **two or more** `Variadic` (or `Optional`) groups needs extra
information to know where one group ends and the next begins. See `AttrSizedOperandSegments` and
`SameVariadicOperandSize` in chapter 7.

---

## 3.6 `Optional`: an operand that may be missing

**The idea.** `Optional<X>` means "zero or one operand of type `X`".

**Example** *(lab op)*:

```tablegen
def Ninja_ExtractOrOp : Ninja_Op<"extract_or", [...]> {
  let arguments = (ins AnyRankedTensor:$input, Optional<AnyType>:$fallback);
  let results = (outs AnyType:$result);
}
```

```mlir
%r = "ninja.extract_or"(%t) : (tensor<4xf32>) -> f32           // no fallback
%r = "ninja.extract_or"(%t, %d) : (tensor<4xf32>, f32) -> f32  // with fallback
```

In C++, `getFallback()` returns a `Value` that is null when the operand is missing:

```cpp
if (Value fb = op.getFallback()) { ... }
```

---

## 3.7 The getter types

The constraint decides the C++ type of the getter. Real examples from the generated code:

| `.td` | Getter returns |
|---|---|
| `F64:$target` | `::mlir::TypedValue<::mlir::FloatType>` |
| `AnyRankedTensor:$input` | `::mlir::TypedValue<::mlir::RankedTensorType>` |
| `AnyType:$x` | `::mlir::TypedValue<::mlir::Type>` |
| `Optional<AnyType>:$fallback` | `::mlir::TypedValue<::mlir::Type>` (null when missing) |
| `Variadic<F64>:$inputs` | `::mlir::Operation::operand_range` |

`TypedValue<FloatType>` is a `Value` that already knows its type is a float, so `.getType()`
gives you a `FloatType` without casting. It converts to a plain `Value` automatically.

---

## Watch out

- **Type rules are only for operands and results.** `AnyTypeOf<[I32, F64Attr]>` mixes a type with
  an attribute and fails with
  `element type 'AttrTypeConstraint' not convertible to 'Type'`.
- **Do not forget the colon:** `AnyFloat:$lhs`, not `AnyFloat$lhs`.
- **`AnyType` does not include `token`.** Producing a token from an `AnyType` result gives
  `result #0 must be any non-token type, but got 'token'`. Use `Token`.
- **Never name an operand after a C++ keyword** (`default`, `class`, `new`, `delete`, ...). ODS
  turns names into C++ parameters, and `Optional<AnyType>:$default` produces C++ that does not
  compile (`expected ',' or '...' before 'default'`).

---

## Exercises

**3.1** Change `AnyFloat` to `F32` in `Ninja_AddOp`, rebuild, and run
`./build/ninja-opt test/ninja-roundtrip.mlir`. Before running, predict the error. Undo the change.

**3.2** Which constraint would you use for: (a) a 2-D tensor of `f32`; (b) either `i32` or
`i64`; (c) any number of `index` values?

**3.3** Add `ninja.sum` from section 3.5 to your ninja. Write a test that sums three values.

**3.4** In `ninja.sum`, what does `getInputs().size()` return for `ninja.sum %a, %b, %c`?

[Solutions](solutions.md#chapter-3)
