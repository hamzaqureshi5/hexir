# 4. Attributes

## In one minute

**What.** An attribute is a **fixed value stored inside an op**: a number, text, a flag, a list, a
type. In `ninja.strike %x : f64 times 3 : f64`, the `3` is an attribute.

**Why.** Many ops need settings that are known when the code is written and never change at run
time: how many times, which direction, what name. If such a setting were an operand (a value
arriving from another op), MLIR could not see it or check it while compiling. As an attribute, it
is right there on the op: passes can read it and verifiers can check it.

**How.** You list attributes in `arguments` with an attribute constraint (`I32Attr`, `StrAttr`,
...). ODS stores them in a C++ struct on the op (the **properties**, shown as `<{...}>` in the
generic form), checks them during verification, and generates two getters for each: one returning
a plain C++ value (`uint32_t getTimes()`) and one returning the attribute object
(`IntegerAttr getTimesAttr()`).

**Where it fits.** Read by the parser, checked by the verifier, used by passes through getters,
written by the printer:
`file.mlir → [PARSE] → [VERIFY] → [PASSES] → [PRINT] → output`

**After this chapter** you can add any kind of fixed value to an op, make it optional or give it a
default, add value rules without C++, and read it correctly in C++.

---

## 4.1 What an attribute is

**The idea.** An attribute is a **fixed value stored on the op**. Unlike an operand, it does not
come from another op. It is known when the IR is written, and it never changes while the program
runs.

**Everyday picture.** A label printed on a product: "Size: M", "Color: red". It is part of the
product; nobody hands it in.

**Syntax.** Attributes go in `(ins ...)` with operands. ODS knows they are attributes because the
constraint ends in `Attr`:

```tablegen
let arguments = (ins F64:$target, I32Attr:$times);
//                   operand      attribute
```

```mlir
%y = ninja.strike %x : f64 times 3 : f64      // 3 is the attribute
```

---

## 4.2 The common attribute constraints

| Constraint | Holds | Written in IR as | C++ getter returns |
|---|---|---|---|
| `I32Attr` | 32-bit integer | `3` | `uint32_t` |
| `SI32Attr` | **signed** 32-bit integer | `-5` | `int32_t` |
| `UI32Attr` | **unsigned** 32-bit integer | `5` | `uint32_t` |
| `I64Attr` | 64-bit integer | `3` | `uint64_t` |
| `F32Attr`, `F64Attr` | float | `2.5` | `llvm::APFloat` |
| `BoolAttr` | true / false | `true` | `bool` |
| `StrAttr` | text | `"left"` | `llvm::StringRef` |
| `UnitAttr` | a flag: present or absent | `fast` | `bool` |
| `TypeAttr` | a type, as data | `f32` | `mlir::Type` |
| `TypeAttrOf<AnyFloat>` | a type that must be a float | `f32` | `mlir::FloatType` |
| `ArrayAttr` | a list of any attributes | `[1, "a"]` | `mlir::ArrayAttr` |
| `I64ArrayAttr`, `StrArrayAttr` | a list of integers / strings | `[1, 2]` | `mlir::ArrayAttr` |
| `DenseI64ArrayAttr` | a compact list of `i64` | `[10, 20]` | `llvm::ArrayRef<int64_t>` |
| `DictionaryAttr` | named attributes | `{a = 1}` | `mlir::DictionaryAttr` |
| `ElementsAttr` | tensor data | `dense<[1, 2]>` | `mlir::ElementsAttr` |
| `SymbolNameAttr` | the name an op **defines** | `@main` | `llvm::StringRef` |
| `FlatSymbolRefAttr` | a name an op **refers to** | `@main` | `llvm::StringRef` |
| `AnyAttr` | any attribute | | `mlir::Attribute` |
| `TypedAttrInterface` | any attribute that carries a type | `3 : i32` | `mlir::TypedAttr` |

**Prefer `DenseI64ArrayAttr` over `I64ArrayAttr`** for a list of integers: it stores them compactly
and gives you a plain `ArrayRef<int64_t>` in C++. In the generic form it prints as
`array<i64: 10, 20>`.

---

## 4.3 Wrappers: optional, default, confined

| Wrapper | Meaning | Getter |
|---|---|---|
| `OptionalAttr<I32Attr>` | may be left out | `std::optional<uint32_t>` |
| `DefaultValuedAttr<I32Attr, "7">` | if left out, it is `7` | `uint32_t` (never missing) |
| `ConfinedAttr<I32Attr, [IntPositive]>` | must also pass extra rules | `uint32_t` |

The default value is a **C++ expression in a string**: `"7"`, `"1.0"`, `"false"`.

---

## 4.4 Example: all of them in one op *(lab op)*

```tablegen
def Ninja_DodgeOp : Ninja_Op<"dodge"> {
  let summary = "dodge in a direction";
  let arguments = (ins
    StrAttr:$direction,                          // required text
    OptionalAttr<I32Attr>:$steps,                // may be left out
    UnitAttr:$fast,                              // a flag
    DefaultValuedAttr<F64Attr, "1.0">:$speed     // 1.0 if left out
  );
  let assemblyFormat =
    "$direction (`steps` $steps^)? (`fast` $fast^)? (`speed` $speed^)? attr-dict";
}
```

(The `( ... )?` groups in the format are explained in chapter 5.)

**What you see.** All accepted:

```mlir
ninja.dodge "left"
ninja.dodge "right" steps 3
ninja.dodge "up" fast
ninja.dodge "down" steps 2 fast speed 4.0
ninja.dodge "back" speed 1.0
```

Printed back, `ninja.dodge "back" speed 1.0` becomes `ninja.dodge "back"`: `1.0` is the default,
so it is not shown.

The generic form shows what is really stored. `speed` is always there (the default is filled in);
`fast` appears only when set:

```mlir
"ninja.dodge"() <{direction = "left", speed = 1.000000e+00 : f64}> : () -> ()
"ninja.dodge"() <{direction = "up", fast, speed = 1.000000e+00 : f64}> : () -> ()
"ninja.dodge"() <{direction = "down", fast, speed = 4.000000e+00 : f64, steps = 2 : i32}> : () -> ()
```

Errors you get for free:

```mlir
"ninja.dodge"() <{speed = 1.0 : f64}> : () -> ()
// error: 'ninja.dodge' op requires attribute 'direction'

ninja.dodge "left" steps 2.5
// error: floating point value not valid for specified type
```

The real getters ODS made:

```cpp
::llvm::StringRef          getDirection();
::std::optional<uint32_t>  getSteps();
bool                       getFast();
::llvm::APFloat            getSpeed();
```

---

## 4.5 `ConfinedAttr`: value rules without C++

**The idea.** `ConfinedAttr<BaseAttr, [rules...]>` takes a normal attribute and adds rules that
the value must pass.

**Example** *(lab op)*:

```tablegen
def Ninja_ShadowOp : Ninja_Op<"shadow"> {
  let arguments = (ins ConfinedAttr<I32Attr, [IntPositive]>:$count);
  let assemblyFormat = "$count attr-dict";
}
```

```mlir
ninja.shadow 5     // ok
ninja.shadow 0
// error: 'ninja.shadow' op attribute 'count' failed to satisfy constraint:
//        32-bit signless integer attribute whose value is positive
ninja.shadow -2    // same error
```

Several rules at once *(lab op)*:

```tablegen
ConfinedAttr<I64Attr, [IntMinValue<1>, IntMaxValue<10>]>:$level
ConfinedAttr<StrArrayAttr, [ArrayMinCount<1>]>:$titles
```

```
ninja.rank 0 ...
error: 'ninja.rank' op attribute 'level' failed to satisfy constraint:
       64-bit signless integer attribute whose minimum value is 1 whose maximum value is 10

ninja.rank 3 scores [] titles [] ...
error: 'ninja.rank' op attribute 'titles' failed to satisfy constraint:
       string array attribute with at least 1 elements
```

**Rules you can use** (all in `mlir/IR/CommonAttrConstraints.td`):

| Rule | Meaning |
|---|---|
| `IntPositive` | > 0 |
| `IntNonNegative` | >= 0 |
| `IntMinValue<n>`, `IntMaxValue<n>` | >= n, <= n |
| `IntPowerOf2` | 1, 2, 4, 8, ... |
| `IntIsOneOf<[1, 2, 4]>` | one of these values |
| `ArrayMinCount<n>`, `ArrayMaxCount<n>`, `ArrayCount<n>` | list length |
| `DenseArrayStrictlyPositive`, `DenseArrayNonNegative` | every element |
| `DenseArraySorted`, `DenseArrayStrictlySorted` | elements in order |

---

## 4.6 Two getters per attribute

Each attribute gets **two** getters:

| Getter | Returns | Example for `I32Attr:$times` |
|---|---|---|
| `getTimes()` | the plain C++ value | `uint32_t` |
| `getTimesAttr()` | the MLIR attribute object | `mlir::IntegerAttr` |

**Everyday picture.** A gift box with `3` inside. `getTimes()` hands you the `3`.
`getTimesAttr()` hands you the box. You can ask "is 3 bigger than 0?", but not "is this box
bigger than 0?"

Use `getTimes()` for logic. Use `getTimesAttr()` when an API wants an `Attribute`, for example
when copying the attribute to a new op.

---

## 4.7 Properties: where attributes live

In the generic form, attributes appear inside `<{ ... }>`:

```mlir
%2 = "ninja.strike"(%1) <{times = 3 : i32}> : (f64) -> f64
```

`<{ }>` means the attribute is stored as a **property**: it is part of the op's own definition,
kept in a typed C++ struct. This is the default and needs nothing from you.

Attributes outside `<{ }>`, in a plain `{ }` dictionary, are **discardable**: extra
information anyone can attach to any op, such as the `device = "cuda"` tags hexir used. A pass is
allowed to drop them.

---

## 4.8 Signed and unsigned: a real bug

`I32Attr` is a **signless** 32-bit integer, and its getter returns `uint32_t`, which cannot be
negative. The starting `ninja.strike` verifier is:

```cpp
if (getTimes() <= 0)   // getTimes() is uint32_t
```

`-1` is stored as the bit pattern `0xFFFFFFFF`. As a `uint32_t` that is 4294967295, which is not
`<= 0`. So this is **accepted**:

```mlir
%y = ninja.strike %x : f64 times -1 : f64       // passes the verifier
```

**Two correct fixes:**

1. `ConfinedAttr<I32Attr, [IntPositive]>`. The rule checks the value as signed, so `-1` and `0`
   are both rejected, as `ninja.shadow` showed.
2. `SI32Attr`, whose getter returns `int32_t`, so `getTimes() <= 0` works. The value is then
   written as `-5 : si32` in the generic form.

Fix 1 is better: no C++ at all.

---

## Watch out

- **The getter's C++ type comes from the constraint**, not from your intention. Check it in
  `NinjaOps.h.inc` before comparing with negative numbers.
- **`DefaultValuedAttr` defaults are C++ code in a string**, so `"1.0"` works but `1.0` (no
  quotes) does not.
- **Every attribute must appear in the assembly format** (chapter 5), or TableGen stops with a
  `missing attribute` error.

---

## Exercises

For hands-on practice with **every** kind of attribute, use
[Practice: every attribute, by hand](practice-attributes.md) alongside these exercises.

**4.1** Prove the bug in section 4.8: create `neg.mlir` with `ninja.strike %x : f64 times -1 : f64`
and run it. Then open `NinjaOps.h.inc` and find the return type of `getTimes()`.

**4.2** Fix it with `ConfinedAttr<I32Attr, [IntPositive]>`, remove `hasVerifier` and
`StrikeOp::verify()`, and update `test/ninja-invalid.mlir` so it expects the new message.

**4.3** Add `ninja.dodge` from section 4.4. Try each line in the "What you see" list, and print
them with `--mlir-print-op-generic`.

**4.4** Which constraint and wrapper would you use for: (a) an optional list of `i64` sizes;
(b) a name that must not be empty; (c) a learning rate that defaults to `0.01`?

[Solutions](solutions.md#chapter-4)
