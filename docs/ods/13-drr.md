# 13. DRR: rewrite rules written in `.td`

## In one minute

**What.** DRR (Declarative Rewrite Rules) lets you write a rewrite as two shapes: "**find** this
pattern of ops, **build** that one instead", in a `.td` file.

**Why.** Many rewrite rules are simple shape changes, like `x + x` → `strike x times 2`. Written as
a C++ pattern class (chapter 10), each takes about 15 lines of matching and building code. As DRR, it
is one or two lines, and harder to get wrong.

**How.** `mlir-tblgen -gen-rewriters` turns each rule into a normal C++ pattern class. You register
it like any other pattern, for example as a canonicalization pattern, and it runs inside the
pass that applies it.

**Where it fits.** In the pass stage:
`file.mlir → PARSE → VERIFY → [PASSES] → PRINT → output`

**After this chapter** you can write simple rewrites in TableGen and know when to use C++
instead.

---

## 13.1 The idea

**DRR** (Declarative Rewrite Rules) describes a rewrite as two pictures: the shape to **find**, and
the shape to **build** instead. TableGen turns each rule into a C++ pattern class, like the
`AddZero` class you wrote by hand in chapter 10.

**Everyday picture.** A find-and-replace rule: find `x + x`, replace it with `strike x times 2`.

**When to use it:** for rules that are just "match this shape of ops, build that shape". When a
rule needs loops, complex conditions or many C++ calls, write a C++ pattern instead.

---

## 13.2 Syntax

DRR rules go in their own file, here `include/ninja/NinjaPatterns.td`:

```tablegen
include "mlir/IR/PatternBase.td"
include "ninja/NinjaOps.td"

// x + x  ==>  strike x times 2
def AddSelfToStrike : Pat<(Ninja_AddOp F64:$x, $x),
                          (Ninja_StrikeOp $x, ConstantAttr<I32Attr, "2">)>;
```

```
Pat< (Ninja_AddOp F64:$x, $x),                       (Ninja_StrikeOp $x, ConstantAttr<I32Attr, "2">) >
      └──────── the shape to FIND ─────────┘          └──────────── the shape to BUILD ────────────────┘
```

| Piece | Meaning |
|---|---|
| `(Ninja_AddOp a, b)` | an `ninja.add` op whose operands match `a` and `b` |
| `$x` | give the matched value a name |
| `$x` used twice | both operands must be **the same** value |
| `F64:$x` | and `x` must have type `f64` |
| `ConstantAttr<I32Attr, "2">` | build a new attribute with value `2` |

The **`F64:$x`** matters: `ninja.strike` only accepts `f64`, so without it the rule would build an
invalid `strike` from an `f32` add.

---

## 13.3 Generating and registering the pattern

**Generate** it in `include/ninja/CMakeLists.txt`:

```cmake
set(LLVM_TARGET_DEFINITIONS NinjaPatterns.td)
mlir_tablegen(NinjaPatterns.inc -gen-rewriters -I ${CMAKE_SOURCE_DIR}/include)
add_public_tablegen_target(NinjaPatternsIncGen)
```

and make the library wait for it in `lib/CMakeLists.txt`:

```cmake
add_dependencies(NinjaDialect NinjaOpsIncGen NinjaPatternsIncGen)
```

**Register** it as a canonicalization pattern on `ninja.add` (`let hasCanonicalizer = 1;`):

```cpp
namespace {
#include "ninja/NinjaPatterns.inc"     // defines struct AddSelfToStrike
} // namespace

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<AddSelfToStrike>(context);
}
```

---

## 13.4 What you see

```mlir
// before --canonicalize
%y = ninja.add %x, %x : f64
%w = ninja.add %v, %v : f32

// after
%y = ninja.strike %x : f64 times 2 : f64     // rewritten
%w = ninja.add %v, %v : f32                  // left alone: F64:$x did not match
```

---

## 13.5 More DRR tools

| Tool | Use |
|---|---|
| `Pattern<find, [build1, build2]>` | build several ops |
| `Pat<find, build, [(Constraint $x)]>` | extra conditions after the match |
| `NativeCodeCall<"C++ expression">` | compute a value in C++ while building |
| `(replaceWithValue $x)` | replace the op with an existing value |
| `ConstantAttr<Type, "value">` | a new constant attribute |
| `$_builder`, `$_loc` | inside `NativeCodeCall`: the builder and location |

For example, "`strike x times 1` becomes `x`" is:

```tablegen
def StrikeOnce : Pat<(Ninja_StrikeOp $x, ConstantAttr<I32Attr, "1">),
                     (replaceWithValue $x)>;
```

The MLIR documentation for all of DRR is in
`/home/user/llvm-project/mlir/docs/DeclarativeRewrites.md`.

---

## Watch out

- **Rules must produce valid IR.** DRR does not check that the built op's types are allowed;
  the verifier catches it after the pass. Add type constraints to the match, as `F64:$x` does.
- **Include path.** `include "ninja/NinjaOps.td"` needs `-I ${CMAKE_SOURCE_DIR}/include` on the
  `mlir_tablegen` line.

---

## Exercises

**13.1** Add `AddSelfToStrike` to your ninja and test it.

**13.2** Remove `F64:` from the rule and try it on an `f32` add. What error do you get, and when?

**13.3** Write `StrikeOnce` from section 13.5 and register it on `ninja.strike`
(`hasCanonicalizer`). Test that `times 1` disappears.

[Solutions](solutions.md#chapter-13)
