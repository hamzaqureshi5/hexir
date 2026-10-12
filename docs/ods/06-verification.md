# 6. Verification

## In one minute

**What.** Verification is MLIR **checking that every op follows its own rules**: the right types,
the required attributes, values in range, sitting in an allowed place, and any rule you add. It
either passes silently or stops with an error that points at the bad op.

**Why.** Every pass is written assuming the IR it receives is valid. If a broken op gets through
(written by hand, or created by a buggy pass), a later pass may compute wrong results or crash far
away from the real cause. Verification catches the problem **immediately, at the op that is
wrong**, with a message saying what is wrong and where.

**How.** For each op, MLIR runs, in order:
1. **structural** trait checks: where the op sits and the shape of its regions (for example
   `HasParent`, `SingleBlock`)
2. checks generated from your `.td`: type constraints, attribute constraints, required attributes
3. checks from the op's other **traits** (for example, "all operand types match")
4. your own C++ check, if you set `let hasVerifier = 1;` and wrote `verify()`

Ops with regions get a second round after the ops inside them are checked. You never call any of
this yourself: MLIR runs it after reading a file, and again after **every** pass
(`--verify-each`, on by default).

**Where it fits.** Right after parsing, and after every pass:
`file.mlir → PARSE → [VERIFY] → PASSES → PRINT → output` (and again after each pass)

**What it cannot do.** It only sees what is known while compiling: types, attributes and the
shape of the IR. It cannot see the value on a wire, because that only exists when the program
runs.

**After this chapter** you know every check MLIR runs and in which order, and you can decide
whether a rule belongs in a constraint, a trait or your own C++.

---

## 6.1 What verification is

**The idea.** Verification is MLIR checking that every op makes sense. It runs after the IR is
read, and after every pass (by default), so a broken op is caught where it was made.


---

## 6.2 The order checks run in

From MLIR's documentation (`mlir/docs/DefiningDialects/Operations.md`, "Verification Ordering"):

**Phase 1, for every op:**

1. **Structural traits** such as `HasParent`, `SingleBlock`, `NoTerminator` (chapter 7).
2. **ODS-generated checks** (`verifyInvariants`): operand and result types (chapter 3), attribute
   types and rules (chapter 4), required attributes.
3. **Other traits and interfaces** with their own checks, such as `SameOperandsAndResultType`.
4. **Your verifier**, if `let hasVerifier = 1;`.

**Phase 2, only for ops with regions, after the ops inside have been verified:**

5. Traits and interfaces that need to look inside regions.
6. **Your region verifier**, if `let hasRegionVerifier = 1;`.

**Why the order matters.** Each step can trust the steps before it. Your `verify()` can call
`getTimes()` without checking that `times` exists or is an `i32`; step 2 already did.

---

## 6.3 Your own verifier: `hasVerifier`

**Syntax:**

```tablegen
def Ninja_StrikeOp : Ninja_Op<"strike", [Pure]> {
  ...
  let hasVerifier = 1;
}
```

ODS declares `::llvm::LogicalResult verify();` on the class. You write it in
`lib/NinjaDialect.cpp`:

```cpp
LogicalResult StrikeOp::verify() {
  if (getTimes() <= 0)
    return emitOpError("times must be positive, got ") << getTimes();
  return success();
}
```

- Return `success()` when the op is fine.
- Return `emitOpError(...)` when it is not. It adds the op name and location for you, and `<<`
  appends more text.

**What you see:**

```mlir
%y = ninja.strike %x : f64 times 0 : f64
// error: 'ninja.strike' op times must be positive, got 0
```

(This particular check has a bug with negative numbers. See chapter 4, section 4.8.)

---

## 6.4 Checking inside regions: `hasRegionVerifier`

**The idea.** If your check needs to look at the ops **inside** your op's region, use
`hasRegionVerifier`. It runs after those inner ops are verified, so you can trust them.

**Example** *(lab op)*: an arena must not be empty.

```tablegen
def Ninja_ArenaOp : Ninja_Op<"arena", [NoTerminator, SingleBlock]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasRegionVerifier = 1;
}
```

```cpp
LogicalResult ArenaOp::verifyRegions() {
  if (getBody().front().empty())
    return emitOpError("arena must contain at least one op");
  return success();
}
```

**What you see:**

```mlir
ninja.arena {
}
// error: 'ninja.arena' op arena must contain at least one op
```

And the order in action: when an op **inside** the arena is broken, that error comes first, and
the arena's own check never runs:

```mlir
ninja.arena {
  %y = "ninja.neg"(%x) : (f64) -> i32
}
// error: 'ninja.neg' op result #0 must be floating-point, but got 'i32'
```

---

## 6.5 Where should a rule go?

Prefer the earliest option that can express the rule:

| Rule | Best place | Chapter |
|---|---|---|
| "this operand must be `f64`" | type constraint | 3 |
| "this attribute must be > 0" | `ConfinedAttr` | 4 |
| "these two values have the same type" | a trait | 7 |
| "this op must sit inside a `ninja.dojo`" | a trait (`HasParent`) | 7 |
| "times must be smaller than the tensor's size" | `hasVerifier` | 6 |
| "the region must contain a `ninja.bow`" | `hasRegionVerifier` | 6 |

**Why.** A constraint or trait is checked by code that thousands of people already use. Your C++
can have bugs: the `getTimes() <= 0` check misses negative numbers because `getTimes()` is
unsigned.

---

## 6.6 What a verifier cannot do

A verifier only sees what is known **while compiling**: types, attributes and the structure of the
IR. It cannot see the value carried by an operand, because that only exists when the program
runs.

```mlir
%n = "some.load"() : () -> i32
%y = ninja.strike_n %x, %n        // nobody knows %n while compiling
```

If `times` were an operand, "times must be positive" could not be verified. It would need a
runtime check in the generated code instead.

---

## Watch out

- **Do not print the op with its custom printer inside a verifier.** Use `emitOpError`. The op is
  not verified yet, so its printer may misbehave; errors print ops in generic form for this
  reason.
- **`hasVerifier` vs `hasRegionVerifier`.** Looking inside regions from `verify()` means looking
  at ops that have **not** been verified yet. Use `verifyRegions()` for that.

---

## Exercises

**6.1** Put these in the order MLIR checks them: your `verify()`; "operand #0 must be 64-bit
float"; `HasParent<"DojoOp">`; your `verifyRegions()`.

**6.2** Add `let hasVerifier = 1;` to `ninja.add` with a rule: both operands must not be the same
SSA value (`getLhs() != getRhs()`). Test with `ninja.add %x, %x : f64`. Then remove it.

**6.3** Rewrite the `ninja.strike` rule so it is correct for negative numbers, **keeping**
`hasVerifier` and `I32Attr`. (Hint: `getTimesAttr()` gives an `IntegerAttr`, which has `getInt()`
returning `int64_t`.)

**6.4** For each rule, choose the best place from section 6.5: (a) a `ninja.kick` op's `power`
attribute must be 1 to 10; (b) `ninja.parry` must be the last op in its block; (c) the number of
results of `ninja.split` must equal its `parts` attribute.

[Solutions](solutions.md#chapter-6)
