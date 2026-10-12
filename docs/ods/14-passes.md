# 14. Declaring passes in `.td`

## In one minute

**What.** A **pass** is one step of a compiler: it walks over the IR and analyses or changes it.
`--canonicalize` and `--cse` are passes. Here you declare your own.

**Why.** A compiler is a sequence of passes. Every pass needs the same plumbing: a command-line
flag, options, a description for `--help`, and registration so tools can find it. TableGen writes
that plumbing from a short declaration, so you only write the part that does the work.

**How.** A `Pass` record in a `.td` file names the flag, what the pass runs on (for example, each
function), and its options. `mlir-tblgen -gen-pass-decls` generates a base class; you inherit from
it and write `runOnOperation()`. A generated `register...Passes()` function makes the pass
available in `ninja-opt`.

**Where it fits.** It **is** the pass stage:
`file.mlir → PARSE → VERIFY → [PASSES] → PRINT → output`

**After this chapter** you can declare and write a pass with options.

---

## 14.1 The idea

A **pass** walks over the IR and does something: analyses, rewrites, lowers. Every pass needs the
same boring parts: a command-line flag, options, a description, registration. TableGen writes
those from a short declaration.

**Everyday picture.** A job advert: title, description, requirements. TableGen prints the advert;
you do the job.

---

## 14.2 Syntax

`include/ninja/Passes.td`:

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

| Piece | Meaning |
|---|---|
| `Pass<"ninja-count-strikes", ...>` | the flag: `--ninja-count-strikes` |
| `"func::FuncOp"` | what the pass runs on: once per function. Leave it out to run on any op |
| `summary` | the line shown in `--help` |
| `Option<"cppName", "flag-name", "C++ type", "default", "help">` | one setting |
| `dependentDialects = [...]` | dialects whose ops the pass creates (they get loaded first) |
| `statistics = [Statistic<...>]` | counters printed with `--mlir-pass-statistics` |

**Build:** `mlir_tablegen(Passes.h.inc -gen-pass-decls -name Ninja)`. `-name Ninja` names the
registration function: `registerNinjaPasses()`.

---

## 14.3 Writing the pass

```cpp
#include "ninja/NinjaDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::ninja {
#define GEN_PASS_DECL                          // declares the options struct
#include "ninja/Passes.h.inc"

#define GEN_PASS_DEF_NINJACOUNTSTRIKES         // defines the base class for this pass
#include "ninja/Passes.h.inc"

namespace {
struct CountStrikes : impl::NinjaCountStrikesBase<CountStrikes> {
  using NinjaCountStrikesBase::NinjaCountStrikesBase;

  void runOnOperation() override {
    int n = 0;
    getOperation().walk([&](StrikeOp) { ++n; });    // getOperation() is the func.func
    if (verbose)                                     // the option, as a member
      llvm::errs() << getOperation().getSymName() << ": ";
    llvm::errs() << n << " strikes\n";
  }
};
} // namespace

#define GEN_PASS_REGISTRATION                    // defines registerNinjaPasses()
#include "ninja/Passes.h.inc"
} // namespace mlir::ninja
```

Then call `mlir::ninja::registerNinjaPasses();` in `ninja-opt`'s `main`, before `MlirOptMain`.

The name in `GEN_PASS_DEF_...` is the record name in capitals: `NinjaCountStrikes` →
`GEN_PASS_DEF_NINJACOUNTSTRIKES`.

---

## 14.4 What you see

```
$ ./build/ninja-opt --help | grep -A1 ninja-count
      --ninja-count-strikes          -   count the ninja.strike ops in each function
        --verbose                    - also print the function name

$ ./build/ninja-opt file.mlir --ninja-count-strikes=verbose=true --mlir-disable-threading -o /dev/null
fold: 0 strikes
strike_once: 1 strikes
strike_twice: 1 strikes
```

`--mlir-disable-threading` is only there to keep the lines in order: passes on `func.func` run on
several functions in parallel.

---

## Watch out

- **You need both `GEN_PASS_DECL` and `GEN_PASS_DEF_<NAME>`.** Without `GEN_PASS_DECL` the build
  fails with `'NinjaCountStrikesOptions' was not declared in this scope`.
- **Link `MLIRPass`** (and `MLIRFuncDialect` for `func::FuncOp`) in `lib/CMakeLists.txt`.
- **A pass may only change what it runs on.** A pass on `func::FuncOp` must not change other
  functions; that is what makes running them in parallel safe.

---

## Exercises

**14.1** Add the `ninja-count-strikes` pass to your ninja and run it on
`test/ninja-roundtrip.mlir`.

**14.2** Add a second option `--ninja-count-strikes=min=2` that prints only functions with at
least `min` strikes. What C++ type and default would you give it?

[Solutions](solutions.md#chapter-14)
