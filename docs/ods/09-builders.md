# 9. Builders

## In one minute

**What.** A builder is a C++ function that **creates a new op**, such as
`StrikeOp::create(builder, loc, x, 3)`.

**Why.** IR is not only read from files. Frontends create it from source code, and passes create
new ops all the time (a rewrite replaces one op with others). Creating an op by hand means filling
in a generic structure: name, operands in the right order, attributes by name, result types. That
is easy to get wrong. A builder takes the op's parts as normal, named C++ parameters.

**How.** ODS generates several `build` functions for every op from its `arguments` and `results`
(for example one taking `uint32_t times`, another taking `IntegerAttr times`). You call them
through `OpName::create(...)`, which also inserts the new op at the builder's current position.
You can add your own builders in the `.td` file for common cases.

**Where it fits.** Inside passes and frontends, whenever new ops are made:
`file.mlir → PARSE → VERIFY → [PASSES] → PRINT → output`

**After this chapter** you can create any op from C++ and add convenient builders of your own.

---

## 9.1 What a builder is

**The idea.** Passes create new ops all the time. A **builder** is the C++ function that creates
one. ODS generates several for every op, from its `arguments` and `results`.

**Everyday picture.** A vending machine button. Press "AddOp", put in two values, and an `AddOp`
comes out.

In this LLVM version you call them as `OpName::create(builder, location, ...)`:

```cpp
auto add = AddOp::create(rewriter, loc, lhs, rhs);
```

- `builder` (often a `rewriter` inside a pattern) decides **where** the new op is inserted.
- `loc` is the source location, used in error messages.
- The rest are the op's operands and attributes, in `arguments` order.

`create` calls the op's `build(...)` functions. When you write your own builder you write a
`build`; callers still use `create`.

---

## 9.2 The builders ODS generates

Real signatures from `build/include/ninja/NinjaOps.h.inc` (namespaces shortened).

**`ninja.strike`** (`(ins F64:$target, I32Attr:$times)`, result `F64`):

```cpp
StrikeOp::create(OpBuilder &, Location, Value target, IntegerAttr times);
StrikeOp::create(OpBuilder &, Location, Value target, uint32_t times);   // plain number
StrikeOp::create(OpBuilder &, Location, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
```

No result type is needed: `F64` is an exact type, so ODS knows it.

**`ninja.add`** (`AnyFloat` operands and result, `SameOperandsAndResultType`):

```cpp
AddOp::create(OpBuilder &, Location, Type result, Value lhs, Value rhs);   // explicit type
AddOp::create(OpBuilder &, Location, Value lhs, Value rhs);                // type from lhs
AddOp::create(OpBuilder &, Location, TypeRange resultTypes, Value lhs, Value rhs);
...
```

`SameOperandsAndResultType` gives the second form: the result type is taken from the operands.
Without that trait, you would have to pass the result type, because `AnyFloat` could be any float.

**The pattern:**

| Kind | Takes |
|---|---|
| separate parameters | one parameter per operand and attribute, in order |
| same, with plain values | attributes as `uint32_t`, `StringRef`, ... instead of `Attribute` |
| collective | `ValueRange operands` + `ArrayRef<NamedAttribute>` (used by generic code) |
| with `Properties` | a `Properties` struct instead of named attributes |

---

## 9.3 Using a builder

Inside `lib/NinjaDialect.cpp` or a pass:

```cpp
// x + x  (result type taken from x)
Value twice = AddOp::create(rewriter, loc, x, x);

// strike x times 3
Value hit = StrikeOp::create(rewriter, loc, x, 3);
```

`create` returns the op. An op with one result converts to its `Value` automatically.

---

## 9.4 Your own builder

**When:** a common way of creating the op needs work that the generated builders do not do, for
example turning a plain C++ value into an attribute.

**Example** *(lab, tested)*: create a constant from a `double`.

```tablegen
def Ninja_ConstantOp : Ninja_Op<"constant", [Pure, ConstantLike]> {
  let arguments = (ins F64Attr:$value);
  let results = (outs F64:$result);
  let builders = [
    OpBuilder<(ins "double":$v), [{
      build($_builder, $_state, $_builder.getF64FloatAttr(v));
    }]>
  ];
}
```

| Piece | Meaning |
|---|---|
| `OpBuilder<(ins ...), [{ body }]>` | one extra builder |
| `"double":$v` | a parameter: C++ type in quotes, then a name |
| `$_builder` | placeholder for the `OpBuilder &` |
| `$_state` | placeholder for the `OperationState &` being filled in |
| `build(...)` inside | calls one of the generated builders |

ODS now also generates:

```cpp
static ConstantOp create(OpBuilder &builder, Location location, double v);
```

so you can write:

```cpp
ConstantOp::create(builder, loc, 2.5);
```

The lab used exactly this in `materializeConstant` (chapter 10), and folding produced
`ninja.constant 3.000000e+00` through it.

**Default arguments.** Use `CArg<"type", "default">`:

```tablegen
OpBuilder<(ins "Value":$x, CArg<"uint32_t", "1">:$times), [{ ... }]>
```

**Body in C++ instead of `.td`.** Leave out the `[{ }]` body and ODS only declares the builder;
you write `void ConstantOp::build(OpBuilder &, OperationState &, double v) { ... }` in the `.cpp`.
hexir's `hextir.for` did this.

---

## 9.5 `skipDefaultBuilders`

```tablegen
let skipDefaultBuilders = 1;
```

Turns off the generated "separate parameter" builders, so only yours exist. Use it when the
generated ones would create invalid ops, for example when building the op also needs a region
with a block in it. hexir's `hextir.prim_func` did this.

---

## Watch out

- **Parameter types are C++ text in quotes:** `"double"`, `"::mlir::Value"`, `"StringRef"`.
- **Builders do not verify.** `create` can build an invalid op; the verifier catches it after the
  pass. Test passes with `--verify-each` (on by default in `ninja-opt`).
- **A terminator used by `SingleBlockImplicitTerminator` needs a builder with no arguments.**
  ODS generates one when the op has no operands or attributes; otherwise add
  `OpBuilder<(ins), [{ }]>`.

---

## Exercises

**9.1** Find all `create` functions for `ConstantOp` in `NinjaOps.h.inc`. Which one would you use
to make the constant `2.5` today?

**9.2** Add the `double` builder from section 9.4 to `ninja.constant`. Check that
`static ConstantOp create(..., double v)` appears in `NinjaOps.h.inc`.

**9.3** Add a builder to `ninja.strike` that takes only `Value target` and always uses `times = 1`.

[Solutions](solutions.md#chapter-9)
