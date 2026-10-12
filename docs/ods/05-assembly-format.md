# 5. Assembly format

## In one minute

**What.** The assembly format is the **text syntax** of an op: what `ninja.strike` looks like in a
`.mlir` file. One line in the `.td` file, `assemblyFormat`, describes it.

**Why.** Every op has a **generic form** (`"ninja.strike"(%1) <{times = 3 : i32}> : (f64) -> f64`)
that always works but is hard to read and write. People read and write IR all the time: in tests,
in bug reports, when debugging passes. A custom syntax makes the IR readable. And because tools
read each other's output, whatever is printed must be readable again (a **round trip**).

**How.** From the one `assemblyFormat` string, ODS generates two C++ functions: a **parser**
(text → op) and a **printer** (op → text). Since both come from the same string, they always
agree. For syntax the string cannot express, you can write one piece (`custom<...>`) or the whole
thing (`hasCustomAssemblyFormat`) in C++.

**Where it fits.** The first and last steps:
`file.mlir → [PARSE] → VERIFY → PASSES → [PRINT] → output`

**After this chapter** you can design an op's text syntax, handle optional parts, and know when
types can be left out.

---

## 5.1 What the assembly format is

**The idea.** `assemblyFormat` is a **template** for how your op is written in a `.mlir` file.
From this one line, ODS writes both the **parser** (text → op) and the **printer** (op → text).

**Everyday picture.** A fill-in-the-blanks sentence: "Strike ___ times ___." The blanks are your
operands and attributes; everything else is fixed text.

**Example from ninja:**

```tablegen
let assemblyFormat = "$target `:` type($target) `times` $times attr-dict `:` type($result)";
```

```
$target  `:`  type($target)  `times`  $times  attr-dict  `:`  type($result)
   │       │        │           │        │        │       │         │
  %x       :       f64        times      3     {extras}   :        f64
```

```mlir
%y = ninja.strike %x : f64 times 3 : f64
```

**The format is a rule for reading too.** Anything in the format must be in your `.mlir` files.
Add `` `:` type($result) `` and every existing `ninja.strike` line without `: f64` stops parsing:

```
error: expected ':'
```

---

## 5.2 The generic form

Every op can also be written in **generic form**, which ignores `assemblyFormat`. It is the op's
true shape:

```mlir
%2 = "ninja.strike"(%1) <{times = 3 : i32}> : (f64) -> f64
//    name in quotes  operands  properties    operand types -> result types
```

Print any file this way:

```bash
./build/ninja-opt test/ninja-roundtrip.mlir --mlir-print-op-generic
```

The generic form always works, even for ops with no `assemblyFormat`. An op without
`assemblyFormat` (and without `hasCustomAssemblyFormat`) can **only** be written in generic form.

---

## 5.3 Every directive, with an example

### `$name`: an operand, attribute or region

```tablegen
"$lhs `,` $rhs attr-dict `:` type($result)"
```
`$lhs` prints the operand, for example `%x`. An attribute prints its value, for example `3`.

### `` `text` ``: fixed text

Backticks mark literal text: keywords like `` `times` ``, punctuation like `` `,` `` `` `:` ``
`` `->` `` `` `(` `` `` `[` ``.

### `attr-dict`: room for extra attributes

**Every format must contain `attr-dict` (or `attr-dict-with-keyword`).** Discardable attributes,
and any attribute not printed elsewhere, are printed here inside `{ }`.

### `attr-dict-with-keyword`

Same as `attr-dict`, but it writes the word `attributes` in front. Use it just before a region, so
the attribute dictionary's `{` is not confused with the region's `{` *(lab op)*:

```tablegen
let assemblyFormat = "attr-dict-with-keyword $body";
```
```mlir
ninja.dojo attributes {open} {
}
```

### `type($x)`: print a type

```tablegen
"$lhs `,` $rhs attr-dict `:` type($result)"
```
```mlir
%0 = ninja.add %a, %b : f32
```

### `qualified(type($x))`: print a type with its full name

Inside an op of the same dialect, MLIR shortens dialect types. With
`` `:` type($star) `` you get `: <4>`; with `` `:` qualified(type($star)) `` you get the full name
*(lab ops)*:

```mlir
%0 = ninja.forge : <4>                     // type($star)
ninja.throw %0 : !ninja.shuriken<4>        // qualified(type($star))
```

### `functional-type(inputs, outputs)`: a function-style signature

```tablegen
let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $result)";
```
```mlir
%m = ninja.mix %a, %i : (f64, i32) -> i32
```

### `operands`, `results`

Shortcuts for "all operands" and "all results", for example `functional-type(operands, results)`.

### `( ... $x^ ... )?`: an optional group

Everything inside the group is printed only when `$x` is present. The `^` marks the **anchor**,
the item that decides. Works with `OptionalAttr`, `UnitAttr`, `DefaultValuedAttr`, `Optional`
operands and `Variadic` operands *(lab op `ninja.dodge`, chapter 4)*:

```tablegen
"$direction (`steps` $steps^)? (`fast` $fast^)? (`speed` $speed^)? attr-dict"
```
```mlir
ninja.dodge "right" steps 3     // steps present: "steps 3" printed
ninja.dodge "left"              // steps absent: the whole group skipped
```

For a `DefaultValuedAttr`, the group is skipped when the value equals the default.

### `$body`: a region

```tablegen
"attr-dict-with-keyword $body"
```
See chapter 11.

### `custom<Name>(...)`: your own C++ for one piece

When one part needs special syntax, write two C++ functions for just that part. Here the count is
written as one `*` per unit *(lab op)*:

```tablegen
def Ninja_StarsOp : Ninja_Op<"stars"> {
  let arguments = (ins I32Attr:$count);
  let assemblyFormat = "custom<Stars>($count) attr-dict";
}
```

ODS calls `parseStars` and `printStars`, which you write in `lib/NinjaDialect.cpp` **before**
`#include "ninja/NinjaOps.cpp.inc"`:

```cpp
static ParseResult parseStars(OpAsmParser &parser, IntegerAttr &count) {
  int32_t n = 0;
  while (succeeded(parser.parseOptionalStar()))
    ++n;
  if (n == 0)
    return parser.emitError(parser.getCurrentLocation(), "expected at least one '*'");
  count = parser.getBuilder().getI32IntegerAttr(n);
  return success();
}

static void printStars(OpAsmPrinter &p, Operation *, IntegerAttr count) {
  for (int64_t i = 0; i < count.getInt(); ++i)
    p << '*';
}
```

```mlir
ninja.stars ***                                  // pretty form
"ninja.stars"() <{count = 3 : i32}> : () -> ()   // generic form
```

The parse function gets a reference to fill in; the print function gets the value plus the
`Operation *`.

### `prop-dict`: all remaining properties in `< >`

Prints every property not already placed in the format as `<name = value, ...>`, without types,
and skips values equal to their default. Useful while you are still deciding on a nicer format.
With `"$direction prop-dict attr-dict"` on `ninja.dodge` *(lab op)*:

```mlir
ninja.dodge "down" <steps = 2, fast>
ninja.dodge "up"                          // nothing to show: no steps, no fast, default speed
ninja.dodge "back" <speed = 3.000000e+00>
```

---

## 5.4 When can a type be left out?

The parser must be able to work out **every** operand and result type. A type can be left out of
the format when ODS can work it out by itself:

| Reason | Example | Written as |
|---|---|---|
| The constraint is one exact type (`F64`, `I32`, `Index`) | `ninja.strike` with `F64` everywhere | `ninja.strike %x times 3` would be enough |
| `SameOperandsAndResultType` | `ninja.add` | `ninja.add %a, %b : f32` (one type) |
| `AllTypesMatch<[...]>` | lab `ninja.select` | `ninja.select %c, %a, %b : f32` |
| `TypesMatchWith<...>` | lab `ninja.extract` | `ninja.extract %t[%i] : tensor<4xf32>` (no result type) |
| `InferTypeOpInterface` (chapter 8) | lab `ninja.elem` | `ninja.elem %t : tensor<4xf32>` |

If ODS cannot work a type out, TableGen stops with an error like:

```
error: type of operand #0, named 'value', is not buildable and a buildable type cannot be inferred
note: suggest adding a type constraint to the operation or adding a 'type($value)' directive to the custom assembly format
```

"Buildable" means "ODS knows how to make this type with no extra information", which is true
for exact types like `F64`.

---

## 5.5 When the template is not enough

Write the whole parser and printer yourself:

```tablegen
let hasCustomAssemblyFormat = 1;
```

ODS then declares two functions for you to write in the `.cpp`:

```cpp
static ParseResult parse(OpAsmParser &parser, OperationState &result);
void print(OpAsmPrinter &p);
```

hexir's `hextir.prim_func` did this, because function signatures (`@name(%a: memref<...>)`) are
too irregular for a template. Prefer `assemblyFormat` plus `custom<>` where possible: it is less
code and handles optional parts for you.

---

## Watch out

- **Every attribute the op declares must appear in the format**, or TableGen stops:
  ```
  error: strict properties in assembly format requires prop-dict unless all inherent attributes
  and properties are bound in the custom assembly format; missing attribute 'speed'
  ```
  Add it (often inside an optional group), or use `prop-dict`.
- **A variadic list at the very end of an op can swallow the next line.** With
  `"$inputs attr-dict"`, an empty `%s = ninja.sum` followed by `%m = ...` on the next line fails
  with `expected operation name in quotes`: the parser reads `%m` as one more input. Put something
  after the list, such as `` `:` type($inputs) ``, or wrap it: `` `(` $inputs `)` ``.
- **Missing `:` before a type.** `"... attr-dict type($result)"` is legal but prints
  `times 3 f64`. MLIR style is `` `:` type(...) ``.

---

## Exercises

**5.1** Change `ninja.strike`'s format to `"$target `times` $times attr-dict `:` type($result)"`.
What do you now have to change in `test/ninja-roundtrip.mlir` and `test/ninja-invalid.mlir`?

**5.2** Write the format for an op written as `ninja.jump %x to %y : f32`, where `%x` and `%y` are
operands and the trait `SameTypeOperands` is used.

**5.3** Add `ninja.dodge` (chapter 4) and remove `` (`speed` $speed^)? `` from its format. Read the
TableGen error. Then fix it with `prop-dict` instead, and look at how the op prints.

**5.4** Write `ninja.stars` from section 5.3. Then change it so zero stars is allowed and prints
nothing.

[Solutions](solutions.md#chapter-5)
