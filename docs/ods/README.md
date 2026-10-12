# Learning ODS and TableGen

A step-by-step guide to writing MLIR operations in `.td` files, using the `ninja` dialect in this
repository as the running example.

## Who this is for

You know what MLIR is: dialects, operations, SSA values, types. You have not written `.td` files
before, or you have copied them without fully understanding them.

## How every chapter is built

Each concept follows the same pattern:

| Part | What it gives you |
|---|---|
| **Goal** | What you will be able to do after the chapter |
| **The idea** | The concept in plain words |
| **Everyday picture** | The same idea, outside of computers |
| **Syntax** | The exact `.td` code |
| **Example** | A working op that uses it |
| **What you see** | Real output from `ninja-opt`: valid IR and the error for invalid IR |
| **Watch out** | Mistakes people often make |
| **Exercises** | Practice. Answers are in [solutions.md](solutions.md) |

## Accuracy

Every example and every error message in this guide was produced by building the op and running
it through `ninja-opt`, built against the LLVM in `/home/user/llvm-project` (LLVM 24.0.0git).
Error messages can change between LLVM versions; the ideas do not.

Some examples use ops that are **not in your ninja yet**. They are marked
*(lab op)*. Their full source is in [lab/](lab/), and the exercises have you add many of them.

## Chapters

**Start here**

| # | Chapter | You learn |
|---|---|---|
| 0 | [The big picture](00-big-picture.md) | what happens to an op from reading to printing, and which `.td` part controls each step |

**Part 1: Foundations** (learn these first, in order)

| # | Chapter | You learn |
|---|---|---|
| 1 | [TableGen basics](01-tablegen.md) | records, classes, `def`, `let`, how `.td` becomes C++ |
| 2 | [Dialects and ops](02-dialects-and-ops.md) | the dialect, the base op class, the four parts of an op |
| 3 | [Type constraints](03-type-constraints.md) | rules for operands and results |
| 4 | [Attributes](04-attributes.md) | fixed values on an op, and their rules |
| 5 | [Assembly format](05-assembly-format.md) | how an op is written as text |
| 6 | [Verification](06-verification.md) | checking that an op is valid |

**Part 2: Traits**

| # | Chapter | You learn |
|---|---|---|
| 7 | [Traits](07-traits.md) | **every** operation trait in MLIR's `OpBase.td`, each with an example |
| 8 | [Interface traits](08-interface-traits.md) | `Pure`, memory effects, speculation, type inference, and more |

**Part 3: Behaviour**

| # | Chapter | You learn |
|---|---|---|
| 9 | [Builders](09-builders.md) | creating ops from C++ |
| 10 | [Folding and canonicalization](10-folding-and-canonicalization.md) | simplifying ops; extra C++ methods |
| 11 | [Regions](11-regions.md) | ops that contain other ops |

**Part 4: Your own vocabulary and tools**

| # | Chapter | You learn |
|---|---|---|
| 12 | [Enums, types and attributes](12-enums-types-attributes.md) | defining your own |
| 13 | [DRR rewrite rules](13-drr.md) | rewrite patterns written in `.td` |
| 14 | [Passes](14-passes.md) | declaring a pass in `.td` |
| 15 | [Advanced TableGen](15-tablegen-advanced.md) | `foreach`, `multiclass`, `!` operators |
| 16 | [Build plumbing and generated code](16-build-and-generated-code.md) | CMake, `.inc` files, docs |

**Practice**

- [Practice: every attribute, by hand](practice-attributes.md): 21 examples and 22 drills
- [Exercises, all in one place](exercises.md)
- [Solutions](solutions.md)

## Tools you will use all the time

```bash
cmake --build build                                      # rebuild after editing .td or .cpp
./build/ninja-opt file.mlir                              # read, check and print
./build/ninja-opt file.mlir --mlir-print-op-generic      # show the true form of each op
./build/ninja-opt file.mlir --canonicalize               # run simplifications
cmake --build build --target check-ninja                 # run the tests
less build/include/ninja/NinjaOps.h.inc                  # the C++ that ODS wrote
```
