# 2. Dialects and ops

## In one minute

**What.** A **dialect** is a named family of ops (`ninja`, `arith`, `func`). An **op** is one kind
of instruction in that family (`ninja.strike`). Every op is made of the same parts: operands,
attributes, results and regions.

**Why.** MLIR has no fixed instruction set. Each project defines its own ops, so MLIR needs a way
to know which ops exist, which family they belong to, and what each one is made of. Without that,
it cannot read, check or print them.

**How.** In the `.td` file, a `Dialect` record describes the family and each `Op` record describes
one op. ODS turns them into C++ classes (`NinjaDialect`, `StrikeOp`, ...). When `ninja-opt` starts,
the dialect is **registered**, and its `initialize()` function tells MLIR about every op. In
memory, each op is a generic `Operation`; `StrikeOp` is a typed view over it with named getters.

**Where it fits.** Everywhere. Every stage relies on the op being registered and on its parts
being known:
`file.mlir → [PARSE] → [VERIFY] → [PASSES] → [PRINT] → output`

**After this chapter** you can define a dialect and ops, and decide whether a value should be an
operand or an attribute.

---

## 2.1 The dialect

**The idea.** A dialect is a **family name** for a group of ops. Every op belongs to exactly one
dialect, and its name starts with the dialect's name: `ninja.add`, `arith.addf`, `func.return`.

**Everyday picture.** A surname. `ninja.add` and `ninja.strike` are in the same family.

**Syntax:**

```tablegen
def Ninja_Dialect : Dialect {
  let name = "ninja";                    // the prefix before the dot
  let cppNamespace = "::mlir::ninja";    // the C++ namespace for everything generated
  let summary = "a toy dialect for learning ODS";
}
```

From this, ODS generates a C++ class called `NinjaDialect`. Its `initialize()` function, which you
write in `lib/NinjaDialect.cpp`, tells MLIR which ops belong to the family:

```cpp
void NinjaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ninja/NinjaOps.cpp.inc"
      >();
}
```

**More dialect fields**, each covered in a later chapter:

| Field | Meaning | Chapter |
|---|---|---|
| `description` | Long documentation | 16 |
| `hasConstantMaterializer = 1` | The dialect can turn a value back into a constant op | 10 |
| `useDefaultTypePrinterParser = 1` | ODS writes the reader/writer for your own types | 12 |
| `useDefaultAttributePrinterParser = 1` | Same, for your own attributes | 12 |
| `dependentDialects = [...]` | Other dialects your code creates ops from | 14 |

**Watch out.** A dialect must also be **registered** in the tool that reads your IR. If it is not,
you get:

```
error: Dialect `ninja' not found for custom op 'ninja.constant'
```

In `tools/ninja-opt/ninja-opt.cpp`:

```cpp
registry.insert<mlir::ninja::NinjaDialect, mlir::func::FuncDialect>();
```

---

## 2.2 The base op class

**The idea.** A small helper class so you do not repeat the dialect on every op.

**Everyday picture.** A rubber stamp with your family name on it.

```tablegen
class Ninja_Op<string mnemonic, list<Trait> traits = []>
    : Op<Ninja_Dialect, mnemonic, traits>;
```

- `mnemonic` is the name after the dot. `"add"` gives `ninja.add`.
- `traits = []` means traits are optional.

Now each op is short:

```tablegen
def Ninja_AddOp : Ninja_Op<"add", [Pure, SameOperandsAndResultType]> { ... }
```

**Naming convention.** Records are usually named `<Dialect>_<Name>Op`, like `Ninja_AddOp`. ODS
drops the prefix for the C++ class, so `Ninja_AddOp` becomes `mlir::ninja::AddOp`.

---

## 2.3 The four parts of an op

**The idea.** An op can have **operands**, **attributes**, **results** and **regions**.

**Everyday picture.** Each op is a **box on a table**, with wires between the boxes.

| Part | Picture | Example |
|---|---|---|
| **Operand** | a wire coming **in** from another box | `%x` |
| **Attribute** | words **printed on** the box | the `3` in `times 3` |
| **Result** | a wire going **out** to other boxes | `%y = ...` |
| **Region** | smaller boxes **inside** the box | `{ ... }` |

```mlir
%x = ninja.constant 2.5
%y = ninja.strike %x : f64 times 3 : f64
```

```
   ┌──────────────────┐
   │ ninja.constant   │
   │   value = 2.5    │  ← attribute (printed on the box)
   └────────┬─────────┘
            │ %x           ← result of constant = operand of strike (a wire)
            ▼
   ┌──────────────────┐
   │ ninja.strike     │
   │   times = 3      │  ← attribute (printed on the box)
   └────────┬─────────┘
            │ %y           ← result (a wire)
            ▼
```

**Syntax:**

```tablegen
let arguments = (ins F64:$target, I32Attr:$times);   // operands AND attributes
let results   = (outs F64:$result);                  // results
let regions   = (region SizedRegion<1>:$body);       // regions (chapter 11)
```

**How ODS tells operands from attributes.** Both go in `(ins ...)`. ODS looks at the constraint in
front of each name:

- `F64` is a **type constraint**, so `$target` is an **operand**.
- `I32Attr` is an **attribute constraint**, so `$times` is an **attribute**.

**Simple rule:** if the constraint name ends in `Attr`, it is an attribute.

---

## 2.4 Operand or attribute: how to choose

| Ask yourself | If yes, use |
|---|---|
| Can the value come from **another op**? | an **operand** (`F64`, `I32`, ...) |
| Is it only known **when the program runs**? | an **operand** |
| Is it a fixed **setting**, known when the code is written? | an **attribute** (`F64Attr`, `I32Attr`, ...) |
| Do you need to **check its value** while compiling? | an **attribute** |

**Everyday picture.** A recipe that says "bake for **20 minutes**" has the number written in it:
an attribute. A recipe that says "bake until the timer says stop" gets the number later: an
operand.

The same `times` can be either:

```mlir
// times as an attribute (I32Attr): the number is part of the op
%y = ninja.strike %x : f64 times 3 : f64

// times as an operand (I32): the number arrives on a wire
%n = arith.constant 3 : i32
%y = ninja.strike %x, %n           // (format would need changing too)
```

With the attribute version, a verifier can check `times > 0` while compiling. With the operand
version it cannot, because the value might only exist at runtime.

---

## 2.5 What ODS generates for one op

For `Ninja_AddOp`, ODS writes a C++ class `AddOp`. The class line in `NinjaOps.h.inc` lists every
trait it has (real output):

```cpp
class AddOp : public ::mlir::Op<AddOp,
    ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult,
    ::mlir::OpTrait::OneTypedResult<::mlir::FloatType>::Impl,
    ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl,
    ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait,
    ::mlir::OpTrait::AlwaysSpeculatableImplTrait,
    ::mlir::MemoryEffectOpInterface::Trait,
    ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait>
```

Notice `ZeroRegions`, `OneResult`, `NOperands<2>`: you never wrote those. ODS adds them from the
shape of `arguments`, `results` and `regions`. Chapter 7 explains every one.

It also generates getters: `getLhs()`, `getRhs()`, `getResult()`, plus builders, a parser, a
printer and a verifier.

---

## Exercises

**2.1** For each part of `%y = ninja.strike %x : f64 times 3 : f64`, say whether it is an operand,
an attribute or a result.

**2.2** Look at the `class StrikeOp` line in `build/include/ninja/NinjaOps.h.inc`. Which traits
did ODS add that you did not write?

**2.3** Comment out `mlir::ninja::NinjaDialect` in the `registry.insert<...>` line of
`tools/ninja-opt/ninja-opt.cpp`, rebuild, and run `./build/ninja-opt test/ninja-roundtrip.mlir`.
Read the error, then undo the change.

**2.4** A `ninja.wait` op pauses for a number of seconds. Should `seconds` be an operand or an
attribute if (a) it is always written in the code, (b) it is computed from user input? Explain.

[Solutions](solutions.md#chapter-2)
