# 1. TableGen basics

## In one minute

**What.** TableGen is a small language for writing down **records**: named bags of fields, like
filled-in forms. A `.td` file is a list of records. ODS is MLIR's set of ready-made record types
(`Dialect`, `Op`, `F64`, `Pure`, ...) written in TableGen.

**Why.** An MLIR op needs hundreds of lines of repetitive C++ (parser, printer, checks, getters,
builders). Writing that by hand for every op is slow and easy to get wrong. A record that
*describes* the op is short, and a tool can turn it into the C++.

**How.** At build time, `mlir-tblgen` reads your `.td` file, works out every record (filling in
everything inherited from parent classes), and hands the records to a **generator** such as
`-gen-op-decls`. The generator writes C++ into `.inc` files, which your `.cpp` and `.h` files
`#include`.

**Where it fits.** Build time only. TableGen never runs when `ninja-opt` runs; only the C++ it
produced does (see [chapter 0](00-big-picture.md#02-two-moments-build-time-and-run-time)).

**After this chapter** you can read any `.td` file's syntax, and see exactly what TableGen made of
it.

---

## 1.1 Why `.td` files exist

**The idea.** Writing an MLIR op by hand in C++ takes hundreds of lines: a class, a parser, a
printer, checks, getters and builders. Most of it is the same for every op. With ODS you
**describe** the op in a few lines, and a tool called `mlir-tblgen` **writes the C++ for you**.

**Everyday picture.** You fill in a short order form, and a factory builds the product. The `.td`
file is the order form. `mlir-tblgen` is the factory. The `.inc` files are the product.

```
NinjaOps.td ──► mlir-tblgen ──► NinjaOps.h.inc / .cpp.inc ──► #include'd into NinjaDialect.h / .cpp
 (you write)     (the tool)       (generated, never edit)         (you write the rest)
```

**Two words you will see a lot:**

- **TableGen** is the *language* `.td` files are written in. LLVM uses it for many things, not
  only MLIR.
- **ODS** (Operation Definition Specification) is MLIR's *set of building blocks* written in
  TableGen: `Dialect`, `Op`, `F64`, `I32Attr`, `Pure`, and so on.

So: TableGen is the grammar, ODS is the vocabulary.

---

## 1.2 Records: `class` and `def`

**The idea.** A TableGen file is a list of **records**. A record is a named bag of fields.

**Everyday picture.** A `class` is a **blank form**. A `def` is a **filled-in form**.

**Syntax:**

```tablegen
class Weapon<string n, int d> {   // a blank form with two things to fill in
  string name = n;                // a field: type, name, value
  int damage = d;
}

def Sword : Weapon<"sword", 10>;  // a filled-in form, built from Weapon
```

| Piece | Meaning |
|---|---|
| `class X<...>` | A blank form. It makes nothing by itself |
| `def Y : X<...>` | A real record named `Y`, built from `X` |
| `<...>` after a class name | Template arguments: the values you must fill in |
| `:` | Inheritance: "built from" |

**What you see.** TableGen can print every record after it fills everything in. This is the real
output for `Sword`:

```
def Sword {	// Weapon
  string name = "sword";
  int damage = 10;
}
```

`// Weapon` is a note saying which classes the record came from.

---

## 1.3 `let`: setting a field

**The idea.** `let` sets a field. It is how you fill in, or change, a box on the form.

```tablegen
def Ninja_AddOp : Ninja_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "add two floats";                        // set the summary field
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs);    // set the arguments field
}
```

Every `let` inside `{ ... }` sets one field of `Ninja_AddOp`. The fields themselves (`summary`,
`arguments`, ...) are declared in MLIR's `Op` class, which `Ninja_Op` inherits from.

---

## 1.4 Values and their types

| Written as | Type | Example |
|---|---|---|
| `"text"` | `string` | `"add two floats"` |
| `[{ ... }]` | `code` (a multi-line string) | `[{ return x; }]` |
| `42` | `int` | `let hasVerifier = 1;` |
| `0` / `1` | `bit` (a yes/no flag) | `let hasFolder = 1;` |
| `[a, b, c]` | `list` | `[Pure, Commutative]` |
| `(ins a, b)` | `dag` | `(ins F64:$x)` |
| `SomeRecord` | a record | `Pure` |

**`dag`** is the unusual one. It is a labelled bundle: `(label item, item, ...)`. ODS uses it for
inputs, outputs and regions:

```tablegen
(ins F64:$target, I32Attr:$times)    // label "ins",  two items
(outs F64:$result)                   // label "outs", one item
```

Each item is `Constraint:$name`. The `$name` part becomes a C++ getter: `$target` becomes
`getTarget()`.

---

## 1.5 `include` and include guards

**The idea.** `include` copies another `.td` file in, so you can use the records it defines.

```tablegen
#ifndef NINJA_OPS        // "if NINJA_OPS is not defined yet..."
#define NINJA_OPS        // "...define it now"

include "mlir/IR/OpBase.td"                         // Dialect, Op, F64, I32Attr, ...
include "mlir/Interfaces/SideEffectInterfaces.td"   // Pure

// ... your records ...

#endif // NINJA_OPS
```

The `#ifndef`/`#define`/`#endif` guard stops a file from being read twice, the same as in C++.

**Watch out.** If you use a name TableGen has never seen, you get:

```
error: Variable not defined: 'SameOperandsAndResultType'
```

It almost always means **a missing `include`**. Find which `.td` file defines the name:

```bash
grep -rln "^def SameOperandsAndResultType" /home/user/llvm-project/mlir/include/mlir/
```

---

## 1.6 Comments

```tablegen
// A line comment.
/* A block comment. */
```

---

## 1.7 Seeing what TableGen made

**`--print-records`** shows every record, fully filled in:

```bash
/home/user/llvm-project/build/bin/mlir-tblgen --print-records \
  -I /home/user/llvm-project/mlir/include include/ninja/NinjaOps.td
```

Real output for `Ninja_AddOp` (shortened):

```
def Ninja_AddOp {	// Op Ninja_Op
  Dialect opDialect = Ninja_Dialect;
  string opName = "add";
  string cppNamespace = "::mlir::ninja";
  string summary = "add two floats";
  dag arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs);
  dag results = (outs AnyFloat:$result);
  string assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  list<Trait> traits = [Pure, SameOperandsAndResultType];
  ...
}
```

You never wrote `opDialect` or `opName` in `Ninja_AddOp`. They come from the classes it is built
from (`Ninja_Op` and MLIR's `Op`). This is the best tool for answering "where did that value come
from?"

**The generated C++** is in `build/include/ninja/`:

```bash
less build/include/ninja/NinjaOps.h.inc
```

---

## Exercises

**1.1** Without running anything: in `def Ninja_StrikeOp : Ninja_Op<"strike", [Pure]> { ... }`,
which part is the record's name, which is the class it is built from, and which are the template
arguments?

**1.2** Run `--print-records` on `include/ninja/NinjaOps.td` and find `Ninja_StrikeOp`. What is
the value of `opName`? Where did it come from?

**1.3** Delete the line `include "mlir/Interfaces/SideEffectInterfaces.td"` from `NinjaOps.td`
and rebuild. Read the error. Which word is "not defined"? Put the line back.

**1.4** Write a tiny `.td` file with a `class Ninja<string n, int rank>` and two `def`s. Print it
with `/home/user/llvm-project/build/bin/llvm-tblgen --print-records your.td`.

[Solutions](solutions.md#chapter-1)
