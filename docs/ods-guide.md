# ODS and TableGen, explained simply

This guide explains every idea you need to write MLIR `.td` files. Each section has:

- **The idea**: one or two plain sentences.
- **Everyday picture**: the same idea, outside of computers.
- **In ninja**: the `.td` code.
- **What you see**: real output from `ninja-opt`.
- **Watch out**: a common mistake, if there is one.

Every output in this guide was produced by running the code, so it is not made up.

Some examples use ops that are **not in your ninja yet**, like `ninja.dodge` or `ninja.dojo`. They are
marked *(not in your ninja yet)*, and the [exercises](#exercises) at the end have you add them.

**How to read it:**

- **Tier 1** (sections 1–11): learn now.
- **Tier 2** (sections 12–17): learn after Tier 1 feels easy.
- **Tier 3** (sections 18–23): learn when you need it.

---

## Part 0: The big picture

**The idea.** Writing an MLIR op by hand in C++ takes hundreds of lines: a class, a parser, a
printer, checks, getters. With ODS you **describe** the op in a short `.td` file, and a tool called
`mlir-tblgen` **writes the C++ for you**.

**Everyday picture.** You fill in a short order form, and a factory builds the product. The `.td`
file is the order form. `mlir-tblgen` is the factory. The `.inc` files are the product.

```
NinjaOps.td ──► mlir-tblgen ──► NinjaOps.h.inc / .cpp.inc ──► #include'd into NinjaDialect.h / .cpp
 (you write)     (the tool)       (generated, never edit)         (you write the rest)
```

**See it yourself.** The generated C++ is here:

```bash
less build/include/ninja/NinjaOps.h.inc
```

**Two words you will see a lot:**

- **TableGen**: the language `.td` files are written in.
- **ODS** (Operation Definition Specification): MLIR's set of building blocks written in TableGen
  (`Dialect`, `Op`, `F64`, `I32Attr`, ...).

---

# Tier 1: Learn now

## 1. The TableGen language

**The idea.** TableGen is a small language for writing down *records*. A record is just a named
bag of fields, like a form with boxes filled in.

**Everyday picture.** A `class` is a **blank form**. A `def` is a **filled-in form**. Inheritance
(`:`) means "start from that blank form, and fill in some boxes".

**The pieces:**

```tablegen
class Weapon<string n, int d> {   // class = blank form. <...> = what you must fill in
  string name = n;                // a field (a box on the form)
  int damage = d;
}

def Sword : Weapon<"sword", 10>;  // def = a real, filled-in form
```

| Piece | Plain meaning |
|---|---|
| `class X<...>` | A blank form. Makes nothing by itself |
| `def Y : X<...>` | A filled-in form, built from `X` |
| `<...>` | The values you pass in, like function arguments |
| `let field = ...;` | Fill in, or change, a box |
| `include "file.td"` | Bring in forms defined in another file |
| `"text"` | A one-line string |
| `[{ ... }]` | A multi-line string (used for code and long descriptions) |
| `[a, b, c]` | A list |
| `(ins a, b)` | A "dag": a labelled bundle. ODS uses it for inputs and outputs |
| `$name` | A name for one item. ODS turns it into a C++ getter |
| `//` | A comment |

**What you see.** You can ask TableGen to show every record after it fills everything in. This is
the real output for `Ninja_AddOp`:

```bash
/home/user/llvm-project/build/bin/mlir-tblgen --print-records \
  -I /home/user/llvm-project/mlir/include include/ninja/NinjaOps.td
```

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

Notice `opDialect` and `opName`: you never wrote those lines in `Ninja_AddOp`. They came from the
blank forms it inherits from (`Ninja_Op` and `Op`).

**Watch out.** If you use a name TableGen has never seen, you get `Variable not defined`. It
almost always means **a missing `include`**. (You hit this with `SameOperandsAndResultType`.)

---

## 2. The dialect

**The idea.** A dialect is a **family name** for your ops. Every op belongs to exactly one dialect.

**Everyday picture.** A surname. `ninja.add` and `ninja.strike` are both in the `ninja` family,
the same way `arith.addf` is in the `arith` family.

**In ninja:**

```tablegen
def Ninja_Dialect : Dialect {
  let name = "ninja";                 // the family name, written before the dot: ninja.add
  let cppNamespace = "::mlir::ninja"; // the C++ namespace: mlir::ninja::AddOp
  let summary = "a toy dialect for learning ODS";
}
```

From this, ODS writes the C++ class `NinjaDialect`. In `lib/NinjaDialect.cpp`, you fill in its
`initialize()` function, which tells MLIR which ops belong to the family.

**Fields you will add later:**

| Field | When you need it |
|---|---|
| `hasConstantMaterializer = 1` | For folding (section 13) |
| `useDefaultTypePrinterParser = 1` | For your own types (section 19) |
| `dependentDialects = [...]` | When your code creates ops from **another** dialect |

**Watch out.** A dialect must be **registered** in the tool, or MLIR does not know it exists. The
error is `Dialect 'ninja' not found`. The registration is in `tools/ninja-opt/ninja-opt.cpp`:
`registry.insert<mlir::ninja::NinjaDialect, ...>()`.

---

## 3. The base op class

**The idea.** A small helper `class` so you do not repeat the same thing on every op.

**Everyday picture.** A rubber stamp with your family name on it. You stamp each new form instead
of writing the name every time.

**In ninja:**

```tablegen
class Ninja_Op<string mnemonic, list<Trait> traits = []>
    : Op<Ninja_Dialect, mnemonic, traits>;
```

Without it, each op would have to say `Op<Ninja_Dialect, "add", [...]>`. With it, you write:

```tablegen
def Ninja_AddOp : Ninja_Op<"add", [Pure, SameOperandsAndResultType]> { ... }
```

- `mnemonic` is the name after the dot: `"add"` gives `ninja.add`.
- `traits = []` means traits are optional; if you give none, you get an empty list.

Every dialect has one of these.

---

## 4. The four parts of an op

**The idea.** An op can have four kinds of things: **operands**, **attributes**, **results** and
**regions**.

**Everyday picture.** Think of each op as a **box on a table**, with wires between the boxes.

- **Operands** are **wires coming in** from other boxes.
- **Attributes** are **words printed on the box**.
- **Results** are **wires going out** to other boxes.
- **Regions** are **smaller boxes inside the box**.

```mlir
%x = ninja.constant 2.5
%y = ninja.strike %x times 3
```

```
   ┌──────────────────┐
   │ ninja.constant   │
   │   value = 2.5    │  ← attribute (printed on the box)
   └────────┬─────────┘
            │ %x           ← result of constant, operand of strike (a wire)
            ▼
   ┌──────────────────┐
   │ ninja.strike     │
   │   times = 3      │  ← attribute (printed on the box)
   └────────┬─────────┘
            │ %y           ← result (a wire)
            ▼
```

**In ninja:**

```tablegen
let arguments = (ins F64:$target, I32Attr:$times);  // operands AND attributes go here
let results   = (outs F64:$result);                 // results go here
let regions   = (region SizedRegion<1>:$body);      // regions go here (not in strike)
```

**How does ODS know `$target` is a wire and `$times` is printed on the box?** They both sit in
`(ins ...)`. ODS looks at the rule in front of the name:

- `F64` is a **type** rule, so `$target` is an **operand** (a wire).
- `I32Attr` is an **attribute** rule, so `$times` is an **attribute** (printed on the box).

**Simple rule.** If the rule ends in `Attr`, it is printed on the box. Otherwise, it is a wire.

**Which one should I use?**

- If the value can come from **another op**, maybe only known when the program runs, use a
  **wire** (`F64`, `I32`, ...).
- If the value is a **fixed setting** known when you write the code, use an **attribute**
  (`F64Attr`, `I32Attr`, ...).

---

## 5. Type constraints (for wires)

**The idea.** Every value in MLIR has a type: `f32`, `f64`, `i32`, `tensor<2x2xf32>`, and so on.
A *type constraint* is a rule on an operand or result that says **which types are allowed**. You
write it once in the `.td` file, and ODS writes the C++ check for you.

**Everyday picture.** A coin slot. It only takes coins. Push a button in, and it does not fit. The
slot is the operand. "Coins only" is the type constraint.

It is also like the type in a C++ function: `double add(double a, double b)` will not take a
string. `AnyFloat:$lhs` means the same for an MLIR op.

**In ninja:**

```tablegen
def Ninja_AddOp : Ninja_Op<"add", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs);   // inputs: any float type
  let results   = (outs AnyFloat:$result);             // output: any float type
}

def Ninja_StrikeOp : Ninja_Op<"strike", [Pure]> {
  let arguments = (ins F64:$target, I32Attr:$times);   // $target must be exactly f64
  let results   = (outs F64:$result);
}
```

The rule goes **before** the colon. The name goes **after** it:

```
AnyFloat : $lhs
   │         └── the name. Gives you getLhs() in C++
   └──────────── the rule. Says which types are allowed
```

**What you see.** Valid IR, which `ninja-opt` accepts:

```mlir
%0 = ninja.add %a, %b : f32      // f32 is a float
%1 = ninja.add %x, %x : f64      // f64 is a float too
%2 = ninja.strike %x times 2     // %x is f64, which is exactly what F64 asks for
```

Invalid IR, rejected with no C++ written by you:

```mlir
%0 = "ninja.add"(%i, %i) : (i32, i32) -> i32
// error: 'ninja.add' op operand #0 must be floating-point, but got 'i32'
//        AnyFloat says "floats only", and i32 is an integer.

%0 = "ninja.strike"(%a) <{times = 2 : i32}> : (f32) -> f64
// error: 'ninja.strike' op operand #0 must be 64-bit float, but got 'f32'
//        F64 means exactly f64. f32 is a float, but not the right one.

%0 = "ninja.add"(%a, %x) : (f32, f64) -> f32
// error: 'ninja.add' op requires the same type for all operands and results
//        Each operand passes AnyFloat on its own, but the
//        SameOperandsAndResultType trait says they must also match each other.
```

That last example shows two different jobs:

- A **constraint** checks **one** value at a time.
- A **trait** (section 7) can check **how values relate to each other**.

These examples use the generic `"ninja.add"(...)` form on purpose. With the pretty form,
`ninja.add %i, %i : i32` fails earlier, while reading the text, with a vaguer message:
`invalid kind of type specified`.

**Strict or loose.** Pick the strictest rule that is still true:

```
AnyType  ⊃  AnyFloat             ⊃  F64
(anything)  (f16, f32, f64, ...)    (only f64)
```

Too loose (`AnyType`) lets bad IR in, which crashes later. Too strict rejects good IR.

**All the common type constraints:**

| Constraint | Accepts |
|---|---|
| `AnyType` | anything |
| `F64`, `F32`, `I32`, `I1`, `Index` | exactly that type |
| `AnyFloat`, `AnyInteger`, `AnySignlessInteger` | any type in that family |
| `AnyTensor`, `AnyMemRef`, `AnyRankedTensor` | tensors or memrefs |
| `TensorOf<[F32, F64]>` | a tensor whose elements are `f32` or `f64` |
| `AnyTypeOf<[F32, I32]>` | `f32` or `i32` |
| `Variadic<F64>` | **zero or more** `f64` wires |
| `Optional<F64>` | **zero or one** `f64` wire |

**Variadic, in action** *(not in your ninja yet)*. `Variadic` means "any number of wires":

```tablegen
def Ninja_SumOp : Ninja_Op<"sum", [Pure]> {
  let arguments = (ins Variadic<F64>:$inputs);
  let results = (outs F64:$result);
  let assemblyFormat = "$inputs attr-dict";
}
```

```mlir
%s = ninja.sum %a, %b, %c        // three wires in, one out
```

In C++, `getInputs()` gives you all of them as a list.

**Watch out.**

- A rule like `AnyTypeOf` is **only for wires**. Putting `AnyTypeOf<[I32, F64Attr]>` in a list
  mixes a type with an attribute, and TableGen fails with
  `element type 'AttrTypeConstraint' not convertible to 'Type'`.
- Do not forget the colon: `AnyFloat:$lhs`, not `AnyFloat$lhs`.

**Try it.** Change `AnyFloat` to `F32` in `Ninja_AddOp`, rebuild, and run
`./build/ninja-opt test/ninja-roundtrip.mlir`. The test uses `f64`. Guess the error before you run
it.

---

## 6. Attribute constraints (for words printed on the box)

**The idea.** An attribute is a **fixed value stored on the op**: a number, a string, a flag. An
*attribute constraint* is a rule that says what kind of fixed value is allowed.

**Everyday picture.** A label printed on a product, like "Size: M" or "Color: red". It is part of
the product. It does not arrive from anywhere else.

**In ninja:**

```tablegen
let arguments = (ins F64:$target, I32Attr:$times);   // $times must be a 32-bit whole number
```

```mlir
%y = ninja.strike %x times 3     // 3 is the attribute
```

**The common attribute constraints:**

| Constraint | Holds | IR example |
|---|---|---|
| `I32Attr`, `I64Attr` | a whole number | `3` |
| `F64Attr`, `F32Attr` | a decimal number | `2.5` |
| `StrAttr` | text | `"left"` |
| `BoolAttr` | true or false | `true` |
| `UnitAttr` | a flag: there or not there | `fast` |
| `TypeAttr` | a type, stored as data | `f32` |
| `ArrayAttr` | a list of attributes | `[1, 2, 3]` |
| `SymbolNameAttr` | a name | `@main` |

**Wrappers that change the rule:**

| Wrapper | Meaning |
|---|---|
| `OptionalAttr<I32Attr>` | may be left out |
| `DefaultValuedAttr<F64Attr, "1.0">` | if left out, it is `1.0` |
| `ConfinedAttr<I32Attr, [IntPositive]>` | must also pass an extra check, here "bigger than 0" |

**All of them in one op** *(not in your ninja yet)*:

```tablegen
def Ninja_DodgeOp : Ninja_Op<"dodge"> {
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

**What you see.** All of these are accepted:

```mlir
ninja.dodge "left"
ninja.dodge "right" steps 3
ninja.dodge "up" fast
ninja.dodge "down" steps 2 fast speed 4.0
ninja.dodge "back" speed 1.0
```

When printed back, `ninja.dodge "back" speed 1.0` becomes just `ninja.dodge "back"`, because
`1.0` is the default, so there is no need to show it.

The generic form shows what is really stored. Notice `speed` is always there (the default is
filled in), and `fast` shows up only when the flag is set:

```mlir
"ninja.dodge"() <{direction = "left", speed = 1.000000e+00 : f64}> : () -> ()
"ninja.dodge"() <{direction = "up", fast, speed = 1.000000e+00 : f64}> : () -> ()
```

Errors you get for free:

```mlir
"ninja.dodge"() <{speed = 1.0 : f64}> : () -> ()
// error: 'ninja.dodge' op requires attribute 'direction'

ninja.dodge "left" steps 2.5
// error: floating point value not valid for specified type
```

**The extra check, `ConfinedAttr`** *(not in your ninja yet)*:

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

No C++ needed for that check.

**The C++ getters.** Each attribute gets a getter that gives you a normal C++ value. These are the
real ones ODS made for `ninja.dodge`:

| `.td` | Getter | You get |
|---|---|---|
| `StrAttr:$direction` | `getDirection()` | `llvm::StringRef` (text) |
| `OptionalAttr<I32Attr>:$steps` | `getSteps()` | `std::optional<uint32_t>` (a number, or nothing) |
| `UnitAttr:$fast` | `getFast()` | `bool` |
| `DefaultValuedAttr<F64Attr, "1.0">:$speed` | `getSpeed()` | `llvm::APFloat` (a decimal) |

There is also a second getter ending in `Attr`, like `getTimesAttr()`. It gives you the MLIR
attribute **object** instead of the plain value.

Think of a gift box with `3` inside. `getTimes()` hands you the `3`. `getTimesAttr()` hands you the
box. You can compare `3` with `0`, but you cannot ask "is this box bigger than 0?"

**Properties.** In the generic form, attributes appear inside `<{ ... }>`. That means they are
stored as **properties**: attributes the op itself declares. This is the default, and you do not
need to do anything. Attributes outside `<{ }>` are extra ones anyone can stick on any op.

**Watch out: `I32Attr` is unsigned in C++.** `getTimes()` returns `uint32_t`, which cannot be
negative. So in your current `StrikeOp::verify()`:

```cpp
if (getTimes() <= 0)   // -1 becomes 4294967295, which is not <= 0
```

`ninja.strike %x times -1` is **accepted**. Try it! That is a real bug in the starting code.
`ConfinedAttr<I32Attr, [IntPositive]>` checks the value as a signed number, so it catches `-1`.
Exercise 2 fixes this.

---

## 7. Traits

**The idea.** A trait is a **promise** about an op. MLIR either checks the promise for you, or
uses it to optimize your code.

**Everyday picture.** Labels on a food package: "vegetarian", "keep frozen". You add the label,
and everyone who handles the package knows how to treat it.

**In ninja.** Traits go in the `[ ]` list:

```tablegen
def Ninja_AddOp : Ninja_Op<"add", [Pure, SameOperandsAndResultType]> { ... }
```

**The traits you will use most**, each with what it really does:

**`Pure`**: "I have no side effects. I only compute a result." So if nobody uses the result, MLIR
deletes the op:

```mlir
// before --canonicalize
%y = ninja.add %x, %x : f64     // %y is never used
// after: the line is gone
```

**`SameOperandsAndResultType`**: "all my inputs and my output have the same type." MLIR checks it
(the error in section 5), and you only write the type once: `ninja.add %a, %b : f64`.

**`Commutative`** *(not in your ninja yet)*: "the order of my inputs does not matter", like
`a + b == b + a`. MLIR uses it to move constants to the right, so similar code looks the same:

```mlir
// before --canonicalize
%k = ninja.constant 5.0
%y = ninja.add %k, %x : f64
// after
%y = ninja.add %x, %k : f64
```

**`Terminator`**: "I must be the last op in a block", like `return`.

**`HasParent<"DojoOp">`**: "I can only live inside a `ninja.dojo`."

```mlir
ninja.yield       // written outside any dojo
// error: 'ninja.yield' op expects parent op 'ninja.dojo'
```

**`SingleBlockImplicitTerminator<"YieldOp">`**: "my region has one block, and the last op in it
is a `ninja.yield`. You may leave it out when writing the text." See section 17.

**`ConstantLike`**: "I am a constant." Folding needs it (section 13).

**`IsolatedFromAbove`**: "the code inside me cannot use wires from outside me." `func.func` has
this trait: a function cannot reach into another function's values.

**`AttrSizedOperandSegments`**: needed when an op has **two or more** `Variadic` or `Optional`
operand groups. With two lists of wires, MLIR needs to know where one list ends and the next
begins.

**Watch out.** A trait only exists if you `include` the `.td` file that defines it **and** include
the matching C++ header in `NinjaDialect.h`. For example, `SameOperandsAndResultType` lives in
`mlir/Interfaces/InferTypeOpInterface.td`, so the header needs
`#include "mlir/Interfaces/InferTypeOpInterface.h"`.

---

## 8. Assembly format: how the op looks as text

**The idea.** `assemblyFormat` is a **template** for how your op is written in a `.mlir` file.
From this one line, ODS writes both the **reader** (text → op) and the **writer** (op → text).

**Everyday picture.** A fill-in-the-blanks sentence: "Strike ___ times ___." The blanks are your
operands and attributes, and the rest is fixed text.

**In ninja:**

```tablegen
let assemblyFormat = "$target `times` $times attr-dict";
```

```
$target   `times`   $times   attr-dict
   │         │         │         └── room for any extra attributes
   │         │         └── the attribute: 3
   │         └── fixed word, written exactly as is
   └── the operand: %x
```

gives

```mlir
ninja.strike %x times 3
```

**Every piece, with an example:**

**`$name`**: print that operand, attribute or region. `$target` prints `%x`.

**`` `word` ``**: fixed text, in backticks. `` `times` `` prints `times`, and `` `,` `` prints a comma.

**`attr-dict`**: room for extra attributes. **Every format must have it.** If someone sticks an
extra attribute on the op, it is printed here in `{ }`.

**`attr-dict-with-keyword`**: the same, but writes the word `attributes` in front. Use it right
before a region, so `{open}` is not confused with the region's `{ }` *(not in your ninja yet)*:

```mlir
ninja.dojo "garden" attributes {open} {
}
```

**`type($x)`**: print the type of `x`. `ninja.add` uses `` `:` type($result) `` to print `: f64`.

**`functional-type(inputs, outputs)`**: print types like a function signature *(not in your ninja yet)*:

```tablegen
let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $result)";
```

```mlir
%m = ninja.mix %a, %i : (f64, i32) -> i32
```

**`( ... $x^ ... )?`**: an **optional group**. Everything inside is printed only when `$x` is
there. The `^` marks which item decides. From `ninja.dodge` in section 6:

```tablegen
(`steps` $steps^)?
```

```mlir
ninja.dodge "right" steps 3   // steps is there, so "steps 3" is printed
ninja.dodge "left"            // steps is not there, so the whole group is skipped
```

**`$body`**: print a region (section 17).

**`custom<Foo>($x)`**: "I will write this piece myself in C++." You write two functions,
`parseFoo` and `printFoo`.

**When the template is not enough**, write the whole reader and writer in C++:

```tablegen
let hasCustomAssemblyFormat = 1;   // then write parse() and print() in the .cpp
```

**What you see.** Every op also has a **generic form** that ignores `assemblyFormat`. It is the
op's true shape:

```bash
./build/ninja-opt test/ninja-roundtrip.mlir --mlir-print-op-generic
```

```mlir
%2 = "ninja.strike"(%1) <{times = 3 : i32}> : (f64) -> f64
```

**Watch out.**

- **Every attribute must appear in the format.** If you forget one, TableGen stops with
  `strict properties in assembly format requires prop-dict unless all inherent attributes and
  properties are bound ...; missing attribute 'speed'`. Fix it by adding the attribute to the
  format, often inside an optional group. Or write `prop-dict`, which prints all of them in `<{ }>`.
- **A variadic list at the very end can eat the next line.** `%s = ninja.sum` with no inputs,
  followed by a line starting with `%m = ...`, makes the reader think `%m` is an input. Put
  something after the list, such as `` `:` type(...) ``, or wrap it in `` `(` $inputs `)` ``.
- **Types from your own dialect get shortened.** `!ninja.shuriken<4>` prints as `<4>` inside a
  ninja op, because the prefix is obvious there.

---

## 9. Verification: is the op valid?

**The idea.** Verification is MLIR checking that an op makes sense, every time the IR is read or
changed.

**Everyday picture.** A ticket inspector. First, the machine checks the obvious things, like a
valid date. Then a person checks the special rules.

**It happens in this order:**

1. **Automatic checks**, free from ODS: operand types, attribute types, required attributes, trait
   rules. All the errors in sections 5–7 are these.
2. **Your own check**, if you ask for one:

   ```tablegen
   let hasVerifier = 1;
   ```

   ODS then expects you to write this function in `lib/NinjaDialect.cpp`:

   ```cpp
   LogicalResult StrikeOp::verify() {
     if (getTimes() <= 0)
       return emitOpError("times must be positive, got ") << getTimes();
     return success();
   }
   ```

   ```mlir
   %y = ninja.strike %x times 0
   // error: 'ninja.strike' op times must be positive, got 0
   ```

3. **Region checks**: `let hasRegionVerifier = 1;` → `verifyRegions()`. It runs after the ops
   inside your regions have been checked.

Because step 1 runs first, your `verify()` can trust that `times` exists and is an `i32`.

**Rule of thumb.** First try to write the rule in the `.td` file, like `ConfinedAttr`. Write C++
only when the rule is too special for that. Less C++ means fewer bugs, as the `-1` bug in
section 6 shows.

**Watch out.** A verifier can only check what is known **while compiling**: attributes and types.
It cannot check the number travelling on a wire, because that is only known when the program runs.

---

## 10. Documentation fields

**The idea.** `summary` and `description` are the op's **manual page**. TableGen can turn them
into a reference document.

**In ninja:**

```tablegen
let summary = "strike a target some number of times";       // one short line
let description = [{
  Multiplies `target` by `times`. ...                        // longer text, Markdown allowed
}];
```

**What you see.** Run the doc generator:

```bash
/home/user/llvm-project/build/bin/mlir-tblgen -gen-op-doc \
  -I /home/user/llvm-project/mlir/include include/ninja/NinjaOps.td
```

Part of the real output:

````
### `ninja.strike` (ninja::StrikeOp)

_Strike a target some number of times_

Syntax:

```
operation ::= `ninja.strike` $target `times` $times attr-dict
```

Multiplies `target` by `times`. ...

#### Attributes:
| times | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### Operands:
| `target` | 64-bit float |
````

This is how the official MLIR dialect docs are made.

---

## 11. Build plumbing: how the pieces connect

**The idea.** CMake runs `mlir-tblgen` several times on your `.td` file, once for each kind of
output. Then your C++ files `#include` the outputs.

**In ninja**, `include/ninja/CMakeLists.txt`:

```cmake
set(LLVM_TARGET_DEFINITIONS NinjaOps.td)
mlir_tablegen(NinjaOps.h.inc -gen-op-decls)            # op classes: declarations
mlir_tablegen(NinjaOps.cpp.inc -gen-op-defs)           # op classes: code
mlir_tablegen(NinjaDialect.h.inc -gen-dialect-decls -dialect=ninja)
mlir_tablegen(NinjaDialect.cpp.inc -gen-dialect-defs -dialect=ninja)
add_public_tablegen_target(NinjaOpsIncGen)             # a name the library can wait for
```

`lib/CMakeLists.txt` has `add_dependencies(NinjaDialect NinjaOpsIncGen)`, so the `.inc` files are
made **before** the C++ that needs them.

**In C++**, one `.inc` file holds several parts. A `#define` picks which part you get:

```cpp
#define GET_OP_CLASSES             // "give me the op classes"
#include "ninja/NinjaOps.cpp.inc"

void NinjaDialect::initialize() {
  addOperations<
#define GET_OP_LIST                // "give me the list of op names"
#include "ninja/NinjaOps.cpp.inc"
      >();
}
```

**Watch out.**

- A new feature in the `.td` file often needs a **new `#include`** in `NinjaDialect.h`. For
  example, `hasCanonicalizeMethod` needs `mlir/IR/PatternMatch.h`, or the build fails with
  `'mlir::PatternRewriter' has not been declared`.
- New MLIR libraries go in `target_link_libraries` in `lib/CMakeLists.txt`, or you get
  `undefined reference` errors when linking.

---

# Tier 2: Learn next

## 12. Builders: creating ops from C++

**The idea.** A builder is a C++ function that **creates** an op. Passes use builders to make new
ops.

**Everyday picture.** A vending machine button. You press "AddOp", put in two values, and an
`AddOp` comes out.

**What ODS gives you for free.** For `ninja.strike`, ODS made `create` functions like these (real
signatures, simplified):

```cpp
StrikeOp::create(builder, loc, target, /*times=*/uint32_t);   // plain number
StrikeOp::create(builder, loc, target, /*times=*/IntegerAttr); // attribute object
```

So in a pass you write:

```cpp
auto strike = StrikeOp::create(rewriter, loc, x, 3);
```

**Your own builder** *(not in your ninja yet)*. Sometimes you want an easier button. This lets
you create a constant from a plain `double`:

```tablegen
def Ninja_ConstantOp : ... {
  let builders = [
    OpBuilder<(ins "double":$v), [{
      build($_builder, $_state, $_builder.getF64FloatAttr(v));
    }]>
  ];
}
```

```cpp
ConstantOp::create(builder, loc, 2.5);   // the new, easy button
```

- `(ins "double":$v)` is what the button takes.
- The `[{ ... }]` is what it does. Here it turns the `double` into an attribute and calls the
  normal builder.
- `$_builder` and `$_state` are placeholders that TableGen fills in for you.

`let skipDefaultBuilders = 1;` turns off the free builders, if you want only your own.

---

## 13. Folding: making one op simpler

**The idea.** Folding means "if this op's answer is already known, replace it with the answer".

**Everyday picture.** Someone writes `1 + 2` on a form. You cross it out and write `3`.

**In ninja** *(not in your ninja yet)*. Three things are needed:

```tablegen
def Ninja_Dialect : Dialect {
  let hasConstantMaterializer = 1;   // 1. the dialect can create a constant op from a value
}
def Ninja_ConstantOp : Ninja_Op<"constant", [Pure, ConstantLike]> {
  let hasFolder = 1;                 // 2. the constant says "my value is ..."
}
def Ninja_AddOp : ... {
  let hasFolder = 1;                 // 3. add says "if both inputs are known, here is the sum"
}
```

And the C++:

```cpp
OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValueAttr(); }

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<FloatAttr>(adaptor.getLhs());  // known value, or null
  auto rhs = dyn_cast_or_null<FloatAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
    return {};                                                // not known: do nothing
  return FloatAttr::get(getType(), lhs.getValueAsDouble() + rhs.getValueAsDouble());
}

Operation *NinjaDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                             Type type, Location loc) {
  return ConstantOp::create(builder, loc, cast<FloatAttr>(value).getValueAsDouble());
}
```

**What you see:**

```mlir
// before: ./build/ninja-opt file.mlir --canonicalize
%a = ninja.constant 1.0
%b = ninja.constant 2.0
%c = ninja.add %a, %b : f64
return %c : f64

// after
%0 = ninja.constant 3.000000e+00
return %0 : f64
```

**Why three pieces?** `AddOp::fold` returns the **number** `3.0`. But the IR needs an **op** that
produces `3.0`. `materializeConstant` turns the number back into a `ninja.constant` op.

---

## 14. Canonicalization: rewrite rules

**The idea.** Canonicalization means rewriting code into a **simpler, standard** shape, using
rules you give. Folding can only replace an op with a known value. Rewrite rules can do more: swap
ops, drop ops, or replace one with another.

**Everyday picture.** A spelling and grammar checker. It knows rules like "x × 1 = x" and applies
them everywhere.

**The simple way: one function** *(not in your ninja yet)*:

```tablegen
def Ninja_StrikeOp : ... {
  let hasCanonicalizeMethod = 1;
}
```

```cpp
// Rule: "strike x times 1" is just x.
LogicalResult StrikeOp::canonicalize(StrikeOp op, PatternRewriter &rewriter) {
  if (op.getTimes() != 1)
    return failure();                         // rule does not apply
  rewriter.replaceOp(op, op.getTarget());     // use x instead
  return success();
}
```

**What you see:**

```mlir
// before
%y = ninja.strike %x times 1
return %y : f64

// after --canonicalize
return %x : f64
```

`times 2` is left alone, because the rule does not apply.

**The bigger way: several pattern classes.** `let hasCanonicalizer = 1;` makes you write
`getCanonicalizationPatterns()`, where you register as many rule classes as you want. Section 21
uses this.

**Watch out.** You need `#include "mlir/IR/PatternMatch.h"` in `NinjaDialect.h` (section 11).

---

## 15. Extra C++ in the generated class

**The idea.** Add your own helper functions to the class ODS generates.

**Everyday picture.** The factory builds the car, and you add a phone holder.

**In ninja** *(not in your ninja yet)*:

```tablegen
def Ninja_StrikeOp : ... {
  let extraClassDeclaration = [{
    /// True when this strike hits only once.
    bool isSingleStrike() { return getTimes() == 1; }
  }];
}
```

Now any C++ code can ask `strike.isSingleStrike()`. The rule from section 14 can become
`if (!op.isSingleStrike()) return failure();`, which is easier to read.

- `extraClassDeclaration` goes in the `.h.inc` (short functions, or just the declaration).
- `extraClassDefinition` goes in the `.cpp.inc` (longer function bodies).

---

## 16. Interfaces: shared abilities

**The idea.** An interface is an **ability** that many different ops can have, so a pass can work
with all of them without knowing each one.

**Everyday picture.** A "USB port". A pass does not care whether it is plugged into a phone or a
camera; it only cares that there is a USB port.

**You already use some.** `Pure` is really a short name for interfaces. The generated docs for
`ninja.strike` (section 10) list them:

```
Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`
```

`NoMemoryEffect` is the ability a pass checks to decide "can I delete this op if unused?"

**Ways to use interfaces:**

- **Ask ODS to declare the methods for you**, then write them in C++:
  `DeclareOpInterfaceMethods<InferTypeOpInterface>` in the traits list.
- **Mark what an operand does to memory**: `Arg<AnyMemRef, "", [MemRead]>:$buf` means "I only
  read this buffer".
- **Make your own ability**:

  ```tablegen
  def Ninja_CostOpInterface : OpInterface<"CostOpInterface"> {
    let methods = [
      InterfaceMethod<"how expensive this op is", "int", "getCost">
    ];
  }
  ```

  This is generated with `-gen-op-interface-decls` / `-gen-op-interface-defs`. Any ninja op can then
  promise "I have a `getCost()`", and a pass can add up costs without knowing which ops exist.

---

## 17. Regions: ops that contain other ops

**The idea.** A region is a **block of code inside an op**. Loops, functions and `if` all have
regions.

**Everyday picture.** A box that contains smaller boxes.

**In ninja** *(not in your ninja yet)*:

```tablegen
def Ninja_DojoOp : Ninja_Op<"dojo", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);   // exactly one block inside
  let assemblyFormat = "$name attr-dict-with-keyword $body";
}

def Ninja_YieldOp : Ninja_Op<"yield", [Pure, Terminator, HasParent<"DojoOp">]> {
  let assemblyFormat = "attr-dict";
}
```

**What you see.** You write:

```mlir
ninja.dojo "hall" {
  %z = ninja.constant 1.0
}
```

The generic form shows the hidden `ninja.yield` that `SingleBlockImplicitTerminator` adds:

```mlir
"ninja.dojo"() <{name = "hall"}> ({
  %3 = "ninja.constant"() <{value = 1.000000e+00 : f64}> : () -> f64
  "ninja.yield"() : () -> ()
}) : () -> ()
```

In the generic form you must write the yield yourself. Without it:

```
error: block with no terminator, has %0 = "ninja.constant"() ...
```

**Region kinds:**

| Rule | Meaning |
|---|---|
| `AnyRegion` | any number of blocks |
| `SizedRegion<1>` | exactly one block |

**Successors** are a related idea for branch ops: `(successor AnySuccessor:$dest)` means "jump to
this block next". You need them only for your own branch op.

---

# Tier 3: Learn when you need it

## 18. Enums: a fixed list of choices

**The idea.** An enum attribute allows only **one value from a fixed list**.

**Everyday picture.** A drop-down menu. You can pick "stealth" or "attack", and nothing else.

**In ninja** *(not in your ninja yet)*:

```tablegen
include "mlir/IR/EnumAttr.td"

def Ninja_Stealth : I32EnumAttrCase<"stealth", 0>;
def Ninja_Attack  : I32EnumAttrCase<"attack", 1>;
def Ninja_StanceAttr : I32EnumAttr<"Stance", "a ninja's stance",
    [Ninja_Stealth, Ninja_Attack]> {
  let cppNamespace = "::mlir::ninja";
}

def Ninja_StanceOp : Ninja_Op<"stance"> {
  let arguments = (ins Ninja_StanceAttr:$stance);
  let assemblyFormat = "$stance attr-dict";
}
```

**What you see:**

```mlir
ninja.stance attack        // ok. Stored as the number 1: <{stance = 1 : i32}>
ninja.stance kick
// error: custom op 'ninja.stance' expected string or keyword containing one of the
//        following enum values for attribute 'stance' [stealth, attack]
```

In C++ you get a real `enum class Stance { stealth, attack }`.

**Build plumbing:** add `-gen-enum-decls` / `-gen-enum-defs` to CMake, and include
`NinjaEnums.h.inc` in the header and `NinjaEnums.cpp.inc` in the `.cpp`.

---

## 19. Your own types

**The idea.** Make a new **type**, like `f64` or `tensor<...>`, but your own.

**Everyday picture.** Inventing a new unit of measure.

**In ninja** *(not in your ninja yet)*:

```tablegen
def Ninja_Dialect : Dialect {
  let useDefaultTypePrinterParser = 1;    // ODS writes the type reader and writer
}

def Ninja_ShurikenType : TypeDef<Ninja_Dialect, "Shuriken"> {
  let mnemonic = "shuriken";
  let summary = "a throwing star with N points";
  let parameters = (ins "unsigned":$points);   // the type carries a number
  let assemblyFormat = "`<` $points `>`";
}
```

**What you see:**

```mlir
%star = ninja.forge : !ninja.shuriken<4>
ninja.throw %star : !ninja.shuriken<4>
```

`!` means "a type from a dialect". You can now use `Ninja_ShurikenType` as a type constraint, for
example `(ins Ninja_ShurikenType:$star)`.

**Build plumbing:** `-gen-typedef-decls` / `-gen-typedef-defs` with `-typedefs-dialect=ninja`,
`#define GET_TYPEDEF_CLASSES` before including the `.inc` files, and
`addTypes<#define GET_TYPEDEF_LIST ...>()` in `initialize()`.

---

## 20. Your own attributes

**The idea.** The same as section 19, but for **attributes**, with `AttrDef` instead of
`TypeDef`. Use it when the built-in attributes (numbers, strings, arrays) cannot describe your data.

```tablegen
def Ninja_RankAttr : AttrDef<Ninja_Dialect, "Rank"> {
  let mnemonic = "rank";
  let parameters = (ins "unsigned":$level, "::llvm::StringRef":$title);
  let assemblyFormat = "`<` $level `,` $title `>`";
}
```

This would give attributes written like `#ninja.rank<3, "jonin">`. `#` means "an attribute from a
dialect", the same way `!` means a type. Build with `-gen-attrdef-decls` / `-gen-attrdef-defs`.

---

## 21. DRR: rewrite rules written in `.td`

**The idea.** DRR (Declarative Rewrite Rules) lets you write a simple rewrite rule as a
"**this shape** becomes **that shape**" picture, instead of C++.

**Everyday picture.** A find-and-replace rule: find `x + x`, replace with `strike x times 2`.

**In ninja** *(not in your ninja yet)*, a separate file, `include/ninja/NinjaPatterns.td`:

```tablegen
include "mlir/IR/PatternBase.td"
include "ninja/NinjaOps.td"

// x + x  ==>  strike x times 2
def AddSelfToStrike : Pat<(Ninja_AddOp F64:$x, $x),
                          (Ninja_StrikeOp $x, ConstantAttr<I32Attr, "2">)>;
```

- The first `( )` is the shape to **find**. Writing `$x` twice means "both inputs are the same
  wire". `F64:$x` means "only when `x` is `f64`", because `strike` only accepts `f64`.
- The second `( )` is what to **build** instead.

Register it as a canonicalization pattern on `AddOp` (with `let hasCanonicalizer = 1;`):

```cpp
namespace {
#include "ninja/NinjaPatterns.inc"     // made by -gen-rewriters
}

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<AddSelfToStrike>(context);
}
```

**What you see:**

```mlir
// before
%y = ninja.add %x, %x : f64      // f64
%w = ninja.add %v, %v : f32      // f32

// after --canonicalize
%y = ninja.strike %x times 2     // rewritten
%w = ninja.add %v, %v : f32      // left alone: F64:$x did not match
```

---

## 22. Passes declared in `.td`

**The idea.** You can declare a **pass** (a tool that walks over the IR and does something) in a
`.td` file. TableGen writes the boring parts: the command-line flag, the options and the
registration.

**In ninja** *(not in your ninja yet)*, `include/ninja/Passes.td`:

```tablegen
include "mlir/Pass/PassBase.td"

def NinjaCountStrikes : Pass<"ninja-count-strikes", "func::FuncOp"> {
  let summary = "count the ninja.strike ops in each function";
  let options = [
    Option<"verbose", "verbose", "bool", /*default=*/"false",
           "also print the function name">
  ];
}
```

- `"ninja-count-strikes"` becomes the flag `--ninja-count-strikes`.
- `"func::FuncOp"` means "run once on each function".
- Each `Option` becomes a setting, like `--ninja-count-strikes=verbose=true`.

You write only the part that does the work:

```cpp
#define GEN_PASS_DECL
#include "ninja/Passes.h.inc"
#define GEN_PASS_DEF_NINJACOUNTSTRIKES
#include "ninja/Passes.h.inc"

struct CountStrikes : impl::NinjaCountStrikesBase<CountStrikes> {
  using NinjaCountStrikesBase::NinjaCountStrikesBase;
  void runOnOperation() override {
    int n = 0;
    getOperation().walk([&](StrikeOp) { ++n; });
    if (verbose)
      llvm::errs() << getOperation().getSymName() << ": ";
    llvm::errs() << n << " strikes\n";
  }
};
```

**What you see:**

```
$ ./build/ninja-opt file.mlir --ninja-count-strikes=verbose=true --mlir-disable-threading
fold: 0 strikes
strike_once: 1 strikes
strike_twice: 1 strikes
```

**Watch out.** You need **both** `GEN_PASS_DECL` and `GEN_PASS_DEF_<NAME>`. Without
`GEN_PASS_DECL`, the build fails with `'NinjaCountStrikesOptions' was not declared in this scope`.
Build with `mlir_tablegen(Passes.h.inc -gen-pass-decls -name Ninja)`.

---

## 23. TableGen shortcuts

**The idea.** Tools for writing **many records with less typing**. You will see them in big
dialects.

```tablegen
class Weapon<string n, int d> {
  string name = n;
  int damage = d;
}

defvar base = 10;                       // defvar: a named constant

def Sword : Weapon<"sword", base>;

foreach w = ["star", "kunai"] in        // foreach: a loop that makes records
  def Throw_ # w : Weapon<!strconcat("throwing ", w), !add(base, 5)>;

multiclass Pair<int d> {                // multiclass: a recipe for several records at once
  def _light : Weapon<NAME # " light", d>;
  def _heavy : Weapon<NAME # " heavy", !mul(d, 2)>;
}
defm Staff : Pair<4>;                   // defm: use the recipe
```

**What you see** (`llvm-tblgen --print-records`):

```
def Staff_heavy  { string name = "Staff heavy";    int damage = 8;  }
def Staff_light  { string name = "Staff light";    int damage = 4;  }
def Sword        { string name = "sword";          int damage = 10; }
def Throw_kunai  { string name = "throwing kunai"; int damage = 15; }
def Throw_star   { string name = "throwing star";  int damage = 15; }
```

| Piece | Meaning |
|---|---|
| `defvar` | a named value |
| `foreach x = [...] in` | repeat for each item |
| `#` | glue names together: `Throw_ # w` → `Throw_star` |
| `!strconcat`, `!add`, `!mul` | built-in functions; they all start with `!` |
| `multiclass` / `defm` | define a group of records once, make it many times |
| `NAME` | inside a `multiclass`, the name given to `defm` (`Staff`) |

---

## Exercises

Do these in order. Together they cover every section above.

| # | Exercise | Sections |
|---|---|---|
| 1 | Add `ninja.sub` (copy `ninja.add`) | 1–8 |
| 2 | Replace `strike`'s C++ verifier with `ConfinedAttr<I32Attr, [IntPositive]>`. Check that `times -1` is now rejected | 6, 9 |
| 3 | Add `ninja.dodge` from section 6 | 6, 8 |
| 4 | Add `ninja.sum` from section 5 | 5, 8 |
| 5 | Add the `double` builder to `ninja.constant` | 12 |
| 6 | Make `ninja.add` of two constants fold | 13 |
| 7 | Rewrite `strike %x times 1` → `%x`, using an `isSingleStrike()` helper | 14, 15 |
| 8 | Add `ninja.dojo` and `ninja.yield` | 7, 17 |
| 9 | Add the `stance` enum | 18 |
| 10 | Add the `!ninja.shuriken<N>` type | 19 |
| 11 | Add the `x + x → strike x times 2` DRR rule | 21 |
| 12 | Add the `--ninja-count-strikes` pass | 22 |

After each one:

```bash
cmake --build build
./build/ninja-opt your-test.mlir
cmake --build build --target check-ninja
```

Then add a test for it in `test/`.
