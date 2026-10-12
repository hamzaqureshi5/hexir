# 10. Folding, canonicalization and extra C++

## In one minute

**What.** **Folding** and **canonicalization** are small, local simplifications that each op
provides for itself: `1.0 + 2.0` becomes `3.0`; `strike x times 1` becomes `x`. The
`--canonicalize` pass applies them all.

**Why.** IR produced by frontends and by other passes is full of small redundancies. Removing them
one pass at a time would be repetitive. Instead, each op says how **it** can be simplified, and one
general pass applies all those rules across the whole program. The result is IR in a standard,
simple shape (its **canonical** form), which makes every later pass easier to write.

**How.** You opt in from the `.td` file:
- `hasFolder`: a `fold()` function that may replace the op with a known value
- `hasCanonicalizeMethod` / `hasCanonicalizer`: rewrite rules that may change the IR more freely

The `--canonicalize` pass collects the folders and rules of every registered op, together with
trait-based simplifications (chapter 7), and applies them again and again until nothing changes
(up to 10 rounds by default).

**Where it fits.** In the pass stage:
`file.mlir → PARSE → VERIFY → [PASSES] → PRINT → output`

**After this chapter** you can make your ops simplify themselves, and add your own C++ helper
methods to op classes.

---

## 10.1 Three ways to simplify

| Way | Can do | You write |
|---|---|---|
| **Folding** (`hasFolder`) | replace **one op** with a known value or an existing value | `fold()` |
| **Canonicalize method** (`hasCanonicalizeMethod`) | any rewrite of **one op**, in one function | `canonicalize()` |
| **Canonicalization patterns** (`hasCanonicalizer`) | any number of rewrite rules | pattern classes |

All three run under `--canonicalize`, together with the trait-based simplifications from
chapter 7 (`Commutative`, `Idempotent`, `Involution`).

**Everyday picture.** Folding is crossing out `1 + 2` and writing `3`. Canonicalization is a
grammar checker applying rules like "x × 1 is just x" everywhere.

---

## 10.2 Folding

**The idea.** "If my answer is already known, replace me with it."

`fold()` returns one of three things:

| Return | Meaning |
|---|---|
| `{}` (empty) | cannot fold; leave the op alone |
| an `Attribute` | the result is this constant value |
| a `Value` | the result is this existing value |

**Example** *(lab, tested)*. Three pieces are needed for constant folding:

```tablegen
def Ninja_Dialect : Dialect {
  ...
  let hasConstantMaterializer = 1;    // 1. turn a value back into a constant op
}

def Ninja_ConstantOp : Ninja_Op<"constant", [Pure, ConstantLike]> {
  ...
  let hasFolder = 1;                  // 2. "my value is ..."
}

def Ninja_AddOp : ... {
  ...
  let hasFolder = 1;                  // 3. "if both inputs are known, here is the sum"
}
```

```cpp
// 2. A constant folds to its own value.
OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValueAttr(); }

// 3. Add two known values.
OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<FloatAttr>(adaptor.getLhs());   // known value, or null
  auto rhs = dyn_cast_or_null<FloatAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
    return {};                                                  // not known: do nothing
  return FloatAttr::get(getType(), lhs.getValueAsDouble() + rhs.getValueAsDouble());
}

// 1. Build a ninja.constant for a folded value.
Operation *NinjaDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                             Type type, Location loc) {
  auto f = dyn_cast<FloatAttr>(value);
  if (!f || !type.isF64())
    return nullptr;
  return ConstantOp::create(builder, loc, f.getValueAsDouble());
}
```

**What the `FoldAdaptor` is.** `adaptor.getLhs()` gives the **constant value** of the `lhs`
operand if it is known (as an `Attribute`), or null if not. `op.getLhs()` would give the `Value`
itself.

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

**Why three pieces?** `AddOp::fold` returns the number `3.0`, but IR needs an **op** that produces
it. `materializeConstant` builds that op.

---

## 10.3 A canonicalize method

**The idea.** One function that may rewrite the op however it likes.

```tablegen
def Ninja_StrikeOp : ... {
  let hasCanonicalizeMethod = 1;
}
```

```cpp
// Rule: "strike x times 1" is just x.
LogicalResult StrikeOp::canonicalize(StrikeOp op, PatternRewriter &rewriter) {
  if (op.getTimes() != 1)
    return failure();                        // rule does not apply
  rewriter.replaceOp(op, op.getTarget());    // use x instead
  return success();
}
```

**Rules for rewrite code:**

- Return `failure()` when the rule does not apply, **without changing anything**.
- Make all changes through the `rewriter` (`replaceOp`, `eraseOp`, `create`), never directly.

**What you see:**

```mlir
// before
%y = ninja.strike %x : f64 times 1 : f64
return %y : f64

// after --canonicalize
return %x : f64
```

`times 2` is left alone.

**Watch out.** Include `mlir/IR/PatternMatch.h` in `NinjaDialect.h`, or the build fails with
`'mlir::PatternRewriter' has not been declared`.

---

## 10.4 Canonicalization patterns

**The idea.** For several rules, or rules that may create other ops, write **pattern classes** and
register them.

```tablegen
def Ninja_AddOp : ... {
  let hasCanonicalizer = 1;
}
```

```cpp
#include "mlir/IR/Matchers.h"

namespace {
/// x + 0.0  ==>  x
struct AddZero : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {
    FloatAttr rhs;
    if (!matchPattern(op.getRhs(), m_Constant(&rhs)) || !rhs.getValue().isZero())
      return failure();
    rewriter.replaceOp(op, op.getLhs());
    return success();
  }
};
} // namespace

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<AddZero>(context);
}
```

`matchPattern(value, m_Constant(&attr))` asks "is this value produced by a constant? If so, give
me its attribute". It works for any op with `ConstantLike` and a folder.

**What you see** (lab `ninja.add` also has `Commutative`):

```mlir
// before --canonicalize
%z = ninja.constant 0.0
%y = ninja.add %x, %z : f64     // x + 0
%w = ninja.add %z, %x : f64     // 0 + x

// after: both become %x
```

`0 + x` works too, without a second rule: `Commutative` first moves the constant to the right,
then `AddZero` matches. Traits and patterns work together.

Chapter 13 shows how to write patterns like this in `.td` instead of C++.

---

## 10.5 Extra C++ methods: `extraClassDeclaration`

**The idea.** Add your own helper methods to the generated op class.

**Everyday picture.** The factory builds the car; you add a phone holder.

```tablegen
def Ninja_StrikeOp : ... {
  let extraClassDeclaration = [{
    /// True when this strike hits only once.
    bool isSingleStrike() { return getTimes() == 1; }
  }];
}
```

Now any C++ can call `strike.isSingleStrike()`, and the canonicalize method reads better:

```cpp
LogicalResult StrikeOp::canonicalize(StrikeOp op, PatternRewriter &rewriter) {
  if (!op.isSingleStrike())
    return failure();
  rewriter.replaceOp(op, op.getTarget());
  return success();
}
```

| Field | Goes into | Use for |
|---|---|---|
| `extraClassDeclaration` | the class in `NinjaOps.h.inc` | short methods, or declarations |
| `extraClassDefinition` | `NinjaOps.cpp.inc` | longer bodies; `$cppClass` means the class name |

---

## Watch out

- **`fold` must not create ops.** It may only return an attribute or an existing value. Rewrites
  that create ops belong in a canonicalization pattern.
- **`fold` sees constants through the adaptor.** `adaptor.getLhs()` is an `Attribute` (or null);
  `getLhs()` is a `Value`.
- **Without `hasConstantMaterializer`**, folding to a new constant fails quietly: the op is left
  as it was.

---

## Exercises

**10.1** Add constant folding to your ninja (`ConstantLike` + `hasFolder` on `constant`,
`hasFolder` on `add`, `hasConstantMaterializer` on the dialect). Test with `1.0 + 2.0`.

**10.2** Add `hasCanonicalizeMethod` to `ninja.strike` so `times 1` disappears. Use an
`isSingleStrike()` helper from `extraClassDeclaration`.

**10.3** Write the `AddZero` pattern. Without `Commutative` on `ninja.add`, does `0 + x`
simplify? Why or why not?

**10.4** Could "`strike x times 2` becomes `add x, x`" be done in `fold`? Why not?

[Solutions](solutions.md#chapter-10)
