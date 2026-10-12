# 12. Enums, your own types and your own attributes

## In one minute

**What.** Ways to add **your own vocabulary** to a dialect: an **enum** (a fixed list of choices,
like `stealth | attack`), a **type** (like `!ninja.shuriken<4>`), and an **attribute kind** (like
`#ninja.rank<3, "jonin">`).

**Why.** Built-in types and attributes (`f32`, `i32`, strings, arrays) do not describe every idea.
You could encode a stance as the number `1` or a string `"attack"`, but then nothing checks it,
and the IR is hard to read. Your own enum, type or attribute is checked by MLIR, printed by name,
and has a proper C++ API.

**How.** Each is a TableGen record (`I32EnumAttr`, `TypeDef`, `AttrDef`) with **parameters** and an
assembly format. ODS generates the C++ class, the parser, the printer, and getters for each
parameter. Types and attributes are **uniqued** in the MLIR context: two `!ninja.shuriken<4>` are
the very same object, which makes comparing them cheap.

**Where it fits.** Read by the parser, checked by the verifier, used by passes, written by the
printer:
`file.mlir → [PARSE] → [VERIFY] → [PASSES] → [PRINT] → output`

**After this chapter** you can add enums, types and attributes to your dialect.

All three examples were built and tested together in the lab.

---

## 12.1 Enums: a fixed list of choices

**The idea.** An enum attribute allows **one value from a fixed list**.

**Everyday picture.** A drop-down menu: you can pick "stealth" or "attack", and nothing else.

**Syntax:**

```tablegen
include "mlir/IR/EnumAttr.td"

def Ninja_Stealth : I32EnumAttrCase<"stealth", 0>;   // name in IR, number stored
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
ninja.stance attack
// generic form: "ninja.stance"() <{stance = 1 : i32}> : () -> ()

ninja.stance kick
// error: custom op 'ninja.stance' expected string or keyword containing one of the
//        following enum values for attribute 'stance' [stealth, attack]
```

**What ODS generates** (real, from `NinjaEnums.h.inc`):

```cpp
enum class Stance : uint32_t {
  stealth = 0,
  attack = 1,
};
::llvm::StringRef stringifyStance(Stance);                 // Stance::attack -> "attack"
::std::optional<Stance> symbolizeStance(::llvm::StringRef); // "attack" -> Stance::attack
```

and on the op, `::mlir::ninja::Stance getStance();`, so C++ code can write
`if (op.getStance() == Stance::attack)`.

**Build plumbing.** In `include/ninja/CMakeLists.txt`:

```cmake
mlir_tablegen(NinjaEnums.h.inc -gen-enum-decls)
mlir_tablegen(NinjaEnums.cpp.inc -gen-enum-defs)
```

Then `#include "ninja/NinjaEnums.h.inc"` in `NinjaDialect.h` (after the dialect include) and
`#include "ninja/NinjaEnums.cpp.inc"` in `NinjaDialect.cpp`.

---

## 12.2 Your own type

**The idea.** A new **type**, used like `f64` or `tensor<...>`. A type can carry **parameters**,
like the `4` in `!ninja.shuriken<4>`.

**Everyday picture.** Inventing a new unit of measure, with a number attached.

**Syntax:**

```tablegen
def Ninja_Dialect : Dialect {
  ...
  let useDefaultTypePrinterParser = 1;   // ODS writes the dialect's type parser/printer
}

class Ninja_Type<string name, string typeMnemonic> : TypeDef<Ninja_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Ninja_ShurikenType : Ninja_Type<"Shuriken", "shuriken"> {
  let summary = "a throwing star with N points";
  let parameters = (ins "unsigned":$points);       // C++ type in quotes, then a name
  let assemblyFormat = "`<` $points `>`";
}
```

| Piece | Meaning |
|---|---|
| `TypeDef<Dialect, "Shuriken">` | C++ class `ShurikenType` |
| `mnemonic` | the name in IR: `!ninja.shuriken` |
| `parameters` | the values each type carries |
| `assemblyFormat` | how the parameters are written, same directives as ops |

**Use it** as a type constraint, like `F64`:

```tablegen
def Ninja_ForgeOp : Ninja_Op<"forge"> {
  let results = (outs Ninja_ShurikenType:$star);
  let assemblyFormat = "attr-dict `:` type($star)";
}
def Ninja_ThrowOp : Ninja_Op<"throw"> {
  let arguments = (ins Ninja_ShurikenType:$star);
  let assemblyFormat = "$star attr-dict `:` qualified(type($star))";
}
```

**What you see:**

```mlir
%star = ninja.forge : !ninja.shuriken<4>
ninja.throw %star : !ninja.shuriken<4>
```

Printed back (real output):

```mlir
%0 = ninja.forge : <4>                      // type($star): prefix shortened
ninja.throw %0 : !ninja.shuriken<4>         // qualified(type($star)): full name
```

**What ODS generates:**

```cpp
static ShurikenType get(::mlir::MLIRContext *context, unsigned points);
unsigned getPoints() const;
```

**Build plumbing:**

```cmake
mlir_tablegen(NinjaTypes.h.inc -gen-typedef-decls -typedefs-dialect=ninja)
mlir_tablegen(NinjaTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ninja)
```

```cpp
// NinjaDialect.h
#define GET_TYPEDEF_CLASSES
#include "ninja/NinjaTypes.h.inc"

// NinjaDialect.cpp
#define GET_TYPEDEF_CLASSES
#include "ninja/NinjaTypes.cpp.inc"

void NinjaDialect::initialize() {
  addOperations<...>();
  addTypes<
#define GET_TYPEDEF_LIST
#include "ninja/NinjaTypes.cpp.inc"
      >();
}
```

`NinjaDialect.cpp` also needs `#include "mlir/IR/DialectImplementation.h"` and
`#include "llvm/ADT/TypeSwitch.h"`, which the generated type parser uses.

---

## 12.3 Your own attribute

**The idea.** The same as a type, but for **attributes**: fixed data with your own structure, when
numbers, strings and arrays are not enough.

**Syntax:**

```tablegen
def Ninja_Dialect : Dialect {
  ...
  let useDefaultAttributePrinterParser = 1;
}

def Ninja_RankAttr : AttrDef<Ninja_Dialect, "Rank"> {
  let mnemonic = "rank";
  let summary = "a ninja's rank: a level and a title";
  let parameters = (ins "unsigned":$level, StringRefParameter<"the title">:$title);
  let assemblyFormat = "`<` $level `,` $title `>`";
}

def Ninja_PromoteOp : Ninja_Op<"promote"> {
  let arguments = (ins Ninja_RankAttr:$rank);
  let assemblyFormat = "$rank attr-dict";
}
```

`StringRefParameter<...>` is needed for text: a plain `"::llvm::StringRef"` parameter would point
at memory that may disappear. `StringRefParameter` makes MLIR keep its own copy.

**What you see:**

```mlir
ninja.promote #ninja.rank<3, "jonin">
// printed back: ninja.promote <3, "jonin">          (shortened inside a ninja op)
// generic form: "ninja.promote"() <{rank = #ninja.rank<3, "jonin">}> : () -> ()

ninja.promote #ninja.rank<3>
// error: expected ','
```

`#` marks a dialect attribute, the same way `!` marks a dialect type.

**What ODS generates:**

```cpp
static RankAttr get(::mlir::MLIRContext *context, unsigned level, ::llvm::StringRef title);
unsigned getLevel() const;
::llvm::StringRef getTitle() const;
```

**Build plumbing:** like types, with `-gen-attrdef-decls` / `-gen-attrdef-defs`
(`-attrdefs-dialect=ninja`), `GET_ATTRDEF_CLASSES`, and `addAttributes<GET_ATTRDEF_LIST>` in
`initialize()`.

---

## Watch out

- **Parameter C++ types go in quotes** (`"unsigned"`), except ready-made parameter classes like
  `StringRefParameter<...>` and `ArrayRefParameter<...>`, which handle memory for you.
- **Forgetting `addTypes<...>` or `addAttributes<...>`** builds fine, then crashes the first time
  the type is used:
  ```
  LLVM ERROR: can't create type 'mlir::ninja::ShurikenType' because storage uniquer isn't
  initialized: the dialect was likely not loaded, or the type wasn't added with addTypes<...>()
  in the Dialect::initialize() method.
  ```
- **Enum case names become C++ names.** Use names that are valid C++ identifiers.

---

## Exercises

**12.1** Add the `Stance` enum and `ninja.stance` to your ninja. In C++, how would you check that
a `StanceOp` is in `attack` stance?

**12.2** Add `!ninja.shuriken<N>`, `ninja.forge` and `ninja.throw`. Make `ninja.throw` print the
type with `qualified(...)`.

**12.3** Add a verifier to `Ninja_ShurikenType` that rejects 0 points. (Hint: `let genVerifyDecl
= 1;` declares `static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
unsigned points);`.)

[Solutions](solutions.md#chapter-12)
