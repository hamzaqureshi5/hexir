# Exercises

Each chapter ends with its own exercises. This page gives the **order** to do them in, a
**capstone project**, and a **self-test** to check you have mastered ODS.

Answers: [solutions.md](solutions.md). Try each exercise before reading its answer.

---

## Recommended order

| Step | Exercises | What you will have built |
|---|---|---|
| 1 | [1.1–1.4](01-tablegen.md#exercises) | You can read records and `--print-records` output |
| 2 | [2.1–2.4](02-dialects-and-ops.md#exercises) | You know the four parts of an op |
| 3 | [3.1–3.4](03-type-constraints.md#exercises) | `ninja.sum` with `Variadic` |
| 4 | [4.1–4.4](04-attributes.md#exercises) | the `times -1` bug fixed; `ninja.dodge` |
| 5 | [5.1–5.4](05-assembly-format.md#exercises) | your own formats, `custom<>` |
| 6 | [6.1–6.4](06-verification.md#exercises) | verifiers, and where each rule belongs |
| 7 | [7.1–7.6](07-traits.md#exercises) | `Commutative`, `Involution`, `select`, segment sizes |
| 8 | [8.1–8.4](08-interface-traits.md#exercises) | memory effects, type inference, result names |
| 9 | [9.1–9.3](09-builders.md#exercises) | custom builders |
| 10 | [10.1–10.4](10-folding-and-canonicalization.md#exercises) | constant folding and rewrite patterns |
| 11 | [11.1–11.4](11-regions.md#exercises) | `ninja.dojo`, `ninja.choice` |
| 12 | [12.1–12.3](12-enums-types-attributes.md#exercises) | an enum, a type, a type verifier |
| 13 | [13.1–13.3](13-drr.md#exercises) | DRR patterns |
| 14 | [14.1–14.2](14-passes.md#exercises) | your first pass |
| 15 | [15.1–15.3](15-tablegen-advanced.md#exercises) | shared base classes, `foreach` |
| 16 | [16.1–16.3](16-build-and-generated-code.md#exercises) | generated code, docs, build order |

After every exercise:

```bash
cmake --build build
./build/ninja-opt your-test.mlir
cmake --build build --target check-ninja
```

and add a lit test in `test/` for what you built. A feature without a test is not finished.

---

## Capstone project: `ninja.train`

Build this from an empty section of `NinjaOps.td`, using only this guide. It uses almost every
chapter.

**The op:** a training session that repeats a block of moves.

```mlir
%score = ninja.train 3 rounds intensity high {
^bb0(%round: index):
  %p = ninja.strike %x : f64 times 2 : f64
  ninja.yield %p : f64
} : f64
```

**Requirements:**

1. `rounds` is an attribute that must be 1–100 (chapter 4).
2. `intensity` is an enum: `low`, `medium`, `high` (chapter 12).
3. The region has one block with one `index` argument, the round number (chapter 11).
4. The region ends with `ninja.yield`, which may only appear inside `ninja.train` (chapter 7).
5. `ninja.yield`'s operand type must match `ninja.train`'s result type (chapter 6,
   `hasRegionVerifier`).
6. The assembly format is exactly as shown above (chapter 5).
7. Design question (chapter 10): when `rounds` is 1, could the canonicalizer replace the op with
   the contents of its region? Write down what makes this harder than removing
   `strike %x times 1`. (Think about the `%round` block argument and the `ninja.yield`.)
8. Tests: one valid file, and one invalid file per rule (chapters 6, 16).

There is no single right answer for the capstone. Compare your design with `scf.for` in
`/home/user/llvm-project/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td`.

---

## Self-test: have you mastered ODS?

Answer without looking anything up. If you can do all of these, you have.

1. Given `%r = ninja.mix %a, %b : (f32, i32) -> f32`, write the `arguments`, `results` and
   `assemblyFormat`.
2. What is the difference between `I32` and `I32Attr`? When would you use each?
3. Name three ways to make sure two operands have the same type. Which one also lets the format
   print the type only once?
4. Why does `ninja.strike %x : f64 times -1 : f64` pass the original verifier? Give two fixes.
5. What does `Pure` expand to? Name an op that should be `NoMemoryEffect` but not `Pure`.
6. What three pieces does constant folding need?
7. `SameVariadicOperandSize` vs `AttrSizedOperandSegments`: which one checks the sizes?
8. Your `.td` fails with `Variable not defined: 'Foo'`. What do you do?
9. Your op prints `%0 = ninja.forge : <4>`. How do you make it print `!ninja.shuriken<4>`?
10. Where do `getLhs()`, `create(...)` and the parser come from, and how do you read them?

Answers are at the end of [solutions.md](solutions.md#self-test).
