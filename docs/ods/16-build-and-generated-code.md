# 16. Build plumbing, generated code and troubleshooting

## In one minute

**What.** The plumbing that connects your `.td` file to a working tool: the CMake rules that run
`mlir-tblgen`, the `.inc` files it produces, and the `#include` lines that pull them into your C++.
Also: how to read the generated code, generate docs, and fix common errors.

**Why.** When something fails, the error often appears in generated code you never wrote, or in
the build system. Knowing what is generated, where it lives, and how it is included turns a
confusing error into a quick fix.

**How.** CMake runs `mlir-tblgen` once per output file (`-gen-op-decls`, `-gen-op-defs`, ...) into
`build/include/ninja/`. A `#define` before each `#include` (such as `GET_OP_CLASSES`) selects which
part of an `.inc` file you get. CMake must also know the C++ depends on the `.inc` files, so they
are generated first.

**Where it fits.** Build time, see [chapter 0](00-big-picture.md#02-two-moments-build-time-and-run-time).

**After this chapter** you can find your way around the generated code and fix the common build
and TableGen errors.

---

## 16.1 From `.td` to a library

**`include/ninja/CMakeLists.txt`** runs `mlir-tblgen` once per output:

```cmake
set(LLVM_TARGET_DEFINITIONS NinjaOps.td)          # the input file
mlir_tablegen(NinjaOps.h.inc -gen-op-decls)       # op classes: declarations
mlir_tablegen(NinjaOps.cpp.inc -gen-op-defs)      # op classes: code
mlir_tablegen(NinjaDialect.h.inc -gen-dialect-decls -dialect=ninja)
mlir_tablegen(NinjaDialect.cpp.inc -gen-dialect-defs -dialect=ninja)
add_public_tablegen_target(NinjaOpsIncGen)        # a name other targets can wait for
```

The outputs land in `build/include/ninja/`, which is on the include path, so C++ includes them as
`"ninja/NinjaOps.h.inc"`.

**`lib/CMakeLists.txt`** builds the C++ and waits for the generated files:

```cmake
add_library(NinjaDialect STATIC NinjaDialect.cpp)
target_link_libraries(NinjaDialect PUBLIC MLIRIR MLIRSideEffectInterfaces MLIRInferTypeOpInterface)
add_dependencies(NinjaDialect NinjaOpsIncGen)     # .inc files first
```

**All the generators used in this guide:**

| Flag | Makes | Chapter |
|---|---|---|
| `-gen-op-decls` / `-gen-op-defs` | op classes | 2 |
| `-gen-dialect-decls` / `-gen-dialect-defs` | the dialect class | 2 |
| `-gen-enum-decls` / `-gen-enum-defs` | enums | 12 |
| `-gen-typedef-decls` / `-gen-typedef-defs` (`-typedefs-dialect=ninja`) | your types | 12 |
| `-gen-attrdef-decls` / `-gen-attrdef-defs` (`-attrdefs-dialect=ninja`) | your attributes | 12 |
| `-gen-op-interface-decls` / `-gen-op-interface-defs` | your interfaces | 8 |
| `-gen-rewriters` | DRR patterns | 13 |
| `-gen-pass-decls -name Ninja` | passes | 14 |
| `-gen-op-doc`, `-gen-dialect-doc -dialect=ninja` | Markdown docs | 16.4 |

---

## 16.2 Including the generated code

One `.inc` file holds several parts. A `#define` before the `#include` picks the part:

```cpp
// NinjaDialect.h
#include "ninja/NinjaDialect.h.inc"      // the dialect class (no #define needed)

#define GET_OP_CLASSES                   // the op class declarations
#include "ninja/NinjaOps.h.inc"
```

```cpp
// NinjaDialect.cpp
#include "ninja/NinjaDialect.cpp.inc"

#define GET_OP_CLASSES                   // the op class code
#include "ninja/NinjaOps.cpp.inc"

void NinjaDialect::initialize() {
  addOperations<
#define GET_OP_LIST                      // a comma-separated list of op classes
#include "ninja/NinjaOps.cpp.inc"
      >();
}
```

| Macro | Gives |
|---|---|
| `GET_OP_CLASSES` | op classes |
| `GET_OP_LIST` | `AddOp, ConstantOp, StrikeOp` |
| `GET_TYPEDEF_CLASSES` / `GET_TYPEDEF_LIST` | type classes / list |
| `GET_ATTRDEF_CLASSES` / `GET_ATTRDEF_LIST` | attribute classes / list |
| `GEN_PASS_DECL`, `GEN_PASS_DEF_<NAME>`, `GEN_PASS_REGISTRATION` | pass parts |

**Order in `NinjaDialect.h`:** the MLIR headers your traits and interfaces need come **first**,
then the dialect, then types/attributes, then ops (ops use the types).

---

## 16.3 A tour of the generated header

`build/include/ninja/NinjaOps.h.inc` has, for each op:

| Part | What it is |
|---|---|
| `class AddOpGenericAdaptorBase` / `AddOpAdaptor` | gives named access to operands **without** an op: used by `fold` (`FoldAdaptor`) and `inferReturnTypes` |
| `struct Properties` | the C++ struct holding the op's attributes (only for ops with attributes) |
| `class AddOp : public ::mlir::Op<AddOp, traits...>` | the op itself, with every trait (chapter 7.2) |
| getters | `getLhs()`, `getTimes()`, `getTimesAttr()`, ... |
| `create(...)` / `build(...)` | builders (chapter 9) |
| `verifyInvariants`, `verify`, `parse`, `print`, `fold`, ... | declared here, defined in `.cpp.inc` or by you |

When something does not compile, **search this file** for the method name. It shows the exact
signature ODS expects you to write.

---

## 16.4 Generating documentation

```bash
mlir-tblgen -gen-dialect-doc -dialect=ninja -I /home/user/llvm-project/mlir/include include/ninja/NinjaOps.td
mlir-tblgen -gen-op-doc                     -I /home/user/llvm-project/mlir/include include/ninja/NinjaOps.td
```

(`mlir-tblgen` is in `/home/user/llvm-project/build/bin/`.)

Real output start:

````
# 'ninja' Dialect

_A toy dialect for learning ODS_

## Operations

### `ninja.add` (ninja::AddOp)
````

For each op it lists the syntax (from `assemblyFormat`), `description`, traits, interfaces,
attributes, operands and results. This is how MLIR's own website docs are made, which is why
`summary` and `description` are worth writing well.

---

## 16.5 Troubleshooting: every error in this guide

| Error | Cause | Fix | Chapter |
|---|---|---|---|
| `Variable not defined: 'X'` | missing `include` in the `.td` | `grep -rln "^def X" .../mlir/include` and include that file | 1 |
| `Dialect 'ninja' not found for custom op` | dialect not registered in the tool | add it to `registry.insert<...>` | 2 |
| `operand #0 must be ..., but got ...` | operand type breaks its constraint | fix the IR, or loosen the constraint | 3 |
| `element type 'AttrTypeConstraint' not convertible to 'Type'` | attribute inside a type list | keep type and attribute rules apart | 3 |
| `expected ',' or '...' before 'default'` (C++) | operand named after a C++ keyword | rename it | 3 |
| `requires attribute 'x'` | required attribute missing | add it, or make it `OptionalAttr` | 4 |
| `failed to satisfy constraint: ... whose value is positive` | `ConfinedAttr` rule broken | fix the value | 4 |
| `missing attribute 'x'` (TableGen) | attribute not in `assemblyFormat` | add it, or use `prop-dict` | 5 |
| `type of operand #0 ... is not buildable` | format cannot work out a type | add `type($x)` or a type-matching trait | 5 |
| `expected ':'` / `expected non-function type` | `.mlir` text does not match the format | update the `.mlir` file | 5 |
| `expected operation name in quotes` after an empty variadic | trailing variadic swallowed the next line | end the list with `:` or wrap it in `( )` | 5 |
| `'PatternRewriter' has not been declared` | header missing | include `mlir/IR/PatternMatch.h` | 10 |
| `'getElementTypeOrSelf' was not declared` | header missing | include `mlir/IR/TypeUtilities.h` | 7 |
| `Scalarizable requires Elementwise to precede it` | trait order | put `Elementwise` first | 7 |
| `static assertion failed: expected operation to take zero operands` | `ConstantLike` on an op with operands | remove the operands or the trait | 7 |
| `has no member named 'getNameAttr'` | `Symbol` without `SymbolName`/`SymbolVisibility` | add both, and a `sym_visibility` attribute | 7 |
| `operand #0 does not dominate this use` in a "graph" op | `HasOnlyGraphRegion` without `RegionKindInterface` | use `GraphRegionNoTerminator` | 7 |
| `'XOptions' was not declared in this scope` | pass without `GEN_PASS_DECL` | add it before `GEN_PASS_DEF_...` | 14 |
| `can't create type ... storage uniquer isn't initialized` (crash) | type not in `addTypes<...>` | add it to `initialize()` | 12 |
| `undefined reference to ...` (link) | MLIR library not linked | add it to `target_link_libraries` | 16 |

---

## Exercises

**16.1** Open `build/include/ninja/NinjaOps.h.inc` and find `class StrikeOpAdaptor`. Which getters
does it have? Why might `fold` want an adaptor instead of the op?

**16.2** Generate the Markdown docs for your dialect with `-gen-dialect-doc` and read the section
for `ninja.strike`. Improve its `description` and generate again.

**16.3** In a **copy** of the project, delete `add_dependencies(NinjaDialect NinjaOpsIncGen)` from
`lib/CMakeLists.txt`, configure a fresh build directory, and build only the library:
`cmake --build <dir> --target NinjaDialect`. What happens, and why?

[Solutions](solutions.md#chapter-16)
