# Practice: every attribute, by hand

This page has two parts:

- **Part A: Examples.** One small op for each kind of attribute. Read them, and try a few.
- **Part B: Drills.** One new op for each kind. **You** write the `.td`. The answer is hidden;
  open it only after you have tried.

Everything here was built and run with the LLVM in `/home/user/llvm-project`. The outputs and error
messages are real.

Chapter 4 ([Attributes](04-attributes.md)) explains the ideas. This page is for practice.

---

## How to practise

For each op:

1. Add the op to `include/ninja/NinjaOps.td`, just before `#endif`.
2. Rebuild: `cmake --build build`
3. Put the IR line into a file `practice.mlir`, inside a function:
   ```mlir
   func.func @f() {
     ninja.kick 3
     return
   }
   ```
4. Run it, and also look at the generic form:
   ```bash
   ./build/ninja-opt practice.mlir
   ./build/ninja-opt practice.mlir --mlir-print-op-generic
   ```
5. Try the **wrong** input too, and read the error.

**Reminder: the three parts of every attribute line.**

```tablegen
let arguments = (ins I32Attr:$power);
//                   ───┬───  ──┬──
//                      │       └─ the name: becomes getPower() in C++
//                      └───────── the kind of attribute
let assemblyFormat = "$power attr-dict";     // where it appears in the text
```

---

# Part A: Examples

## A1. Whole numbers

### `I32Attr`: a 32-bit whole number

```tablegen
def Ninja_KickOp : Ninja_Op<"kick"> {
  let arguments = (ins I32Attr:$power);
  let assemblyFormat = "$power attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.kick 3` |
| Printed back | `ninja.kick 3` |
| Stored (generic) | `"ninja.kick"() <{power = 3 : i32}> : () -> ()` |
| C++ getter | `uint32_t getPower();` |
| Wrong | `ninja.kick 2.5` → `error: floating point value not valid for specified type` |
| Wrong | `ninja.kick 5000000000` → `error: integer constant out of range for attribute` |
| Missing | `"ninja.kick"() : () -> ()` → `error: 'ninja.kick' op requires attribute 'power'` |

Note the getter is `uint32_t`: it cannot be negative in C++ (chapter 4, section 4.8).

### `I64Attr`: a 64-bit whole number

```tablegen
def Ninja_RunOp : Ninja_Op<"run"> {
  let arguments = (ins I64Attr:$meters);
  let assemblyFormat = "$meters attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.run 5000` |
| Stored | `<{meters = 5000 : i64}>` |
| C++ getter | `uint64_t getMeters();` |
| Wrong | `ninja.run "far"` → `error: custom op 'ninja.run' invalid kind of attribute specified` |

### `SI32Attr`: a **signed** 32-bit number (can be negative)

```tablegen
def Ninja_StepOp : Ninja_Op<"step"> {
  let arguments = (ins SI32Attr:$offset);
  let assemblyFormat = "$offset attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.step -2` |
| Stored | `<{offset = -2 : si32}>` (note `si32`, not `i32`) |
| C++ getter | `int32_t getOffset();` (signed) |

### `UI32Attr`: an **unsigned** 32-bit number (never negative)

```tablegen
def Ninja_CountOp : Ninja_Op<"count"> {
  let arguments = (ins UI32Attr:$amount);
  let assemblyFormat = "$amount attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.count 7` |
| Stored | `<{amount = 7 : ui32}>` |
| C++ getter | `uint32_t getAmount();` |
| Wrong | `ninja.count -1` → `error: negative integer literal not valid for unsigned integer type` |

**`I32Attr` vs `SI32Attr` vs `UI32Attr`:** all hold 32 bits. Only `SI32Attr` gives a signed
getter, and only `UI32Attr` rejects `-1` in the text.

---

## A2. Decimal numbers

### `F32Attr` (and `F64Attr`, `F16Attr`)

```tablegen
def Ninja_AimOp : Ninja_Op<"aim"> {
  let arguments = (ins F32Attr:$accuracy);
  let assemblyFormat = "$accuracy attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.aim 0.5` |
| Printed back | `ninja.aim 5.000000e-01` (MLIR always prints floats this way) |
| Stored | `<{accuracy = 5.000000e-01 : f32}>` |
| C++ getter | `::llvm::APFloat getAccuracy();` (use `.convertToFloat()` for a `float`) |
| Wrong | `ninja.aim 3` → `error: unexpected decimal integer literal for a floating point value` |

Write `3.0`, not `3`, for a float attribute.

---

## A3. Yes/no, and flags

### `BoolAttr`: `true` or `false`, always written

```tablegen
def Ninja_HideOp : Ninja_Op<"hide"> {
  let arguments = (ins BoolAttr:$visible);
  let assemblyFormat = "$visible attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.hide true` |
| Stored | `<{visible = true}>` |
| C++ getter | `bool getVisible();` |
| Wrong | `ninja.hide yes` → `error: expected attribute value` |

### `UnitAttr`: a flag that is there or not there

```tablegen
def Ninja_SneakOp : Ninja_Op<"sneak"> {
  let arguments = (ins UnitAttr:$quiet);
  let assemblyFormat = "(`quiet` $quiet^)? attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.sneak` or `ninja.sneak quiet` |
| Stored | nothing, or `<{quiet}>` |
| C++ getter | `bool getQuiet();` (true when the word is there) |
| Wrong | `ninja.sneak loud` → `error: custom op 'loud' is unknown (tried 'func.loud' as well)` |

A `UnitAttr` needs an **optional group** in the format (chapter 5): `` (`quiet` $quiet^)? ``.
In the wrong example, `loud` is not the flag word, so MLIR thinks it is the start of the next op.

**`BoolAttr` vs `UnitAttr`:** `BoolAttr` is always written (`true`/`false`); `UnitAttr` is a word
that is either present or absent.

---

## A4. Text

### `StrAttr`

```tablegen
def Ninja_SayOp : Ninja_Op<"say"> {
  let arguments = (ins StrAttr:$words);
  let assemblyFormat = "$words attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.say "hello"` |
| Stored | `<{words = "hello"}>` |
| C++ getter | `::llvm::StringRef getWords();` |
| Wrong | `ninja.say hello` → `error: expected attribute value` (quotes are required) |

---

## A5. A type, stored as data

### `TypeAttr`

```tablegen
def Ninja_ForgeAsOp : Ninja_Op<"forge_as"> {
  let arguments = (ins TypeAttr:$kind);
  let assemblyFormat = "$kind attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.forge_as f32` or `ninja.forge_as tensor<2x3xi8>` |
| Stored | `<{kind = f32}>` |
| C++ getter | `::mlir::Type getKind();` |
| Wrong | `ninja.forge_as 3` → `error: integer literal not valid for specified type` |

Use `TypeAttrOf<AnyFloat>` to allow only some types; its getter returns `::mlir::FloatType`.

---

## A6. Lists

### `ArrayAttr`: a list of **any** attributes

```tablegen
def Ninja_BagOp : Ninja_Op<"bag"> {
  let arguments = (ins ArrayAttr:$items);
  let assemblyFormat = "$items attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.bag [1, "rope", 2.5, true]` |
| Printed back | `ninja.bag [1, "rope", 2.500000e+00, true]` |
| C++ getter | `::mlir::ArrayAttr getItems();` (loop over it; each item is an `Attribute`) |
| Wrong | `ninja.bag 1` → `error: integer literal not valid for specified type` |

### `StrArrayAttr` (and `I64ArrayAttr`, `I32ArrayAttr`, ...): a list of **one** kind

```tablegen
def Ninja_TeamOp : Ninja_Op<"team"> {
  let arguments = (ins StrArrayAttr:$members);
  let assemblyFormat = "$members attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.team ["kenji", "aiko"]` |
| C++ getter | `::mlir::ArrayAttr getMembers();` |
| Wrong | `ninja.team [1, 2]` → `error: 'ninja.team' op attribute 'members' failed to satisfy constraint: string array attribute` |

### `DenseI64ArrayAttr` (and `DenseI32ArrayAttr`, `DenseF32ArrayAttr`, ...): a compact list of numbers

```tablegen
def Ninja_PathOp : Ninja_Op<"path"> {
  let arguments = (ins DenseI64ArrayAttr:$steps);
  let assemblyFormat = "$steps attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.path [1, 2, 3]` |
| Stored | `<{steps = array<i64: 1, 2, 3>}>` |
| C++ getter | `::llvm::ArrayRef<int64_t> getSteps();` (plain numbers: easy to loop over) |
| Wrong | `ninja.path [1.5]` → `error: custom op 'ninja.path' expected integer value` |

**For a list of numbers, prefer `DenseI64ArrayAttr` over `I64ArrayAttr`:** it is smaller, and its
getter gives you plain `int64_t`s instead of attribute objects.

---

## A7. Named values

### `DictionaryAttr`: name = value pairs

```tablegen
def Ninja_ProfileOp : Ninja_Op<"profile"> {
  let arguments = (ins DictionaryAttr:$info);
  let assemblyFormat = "$info attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.profile {age = 20 : i32, clan = "leaf"}` |
| C++ getter | `::mlir::DictionaryAttr getInfo();` (use `.get("age")`) |
| Wrong | `ninja.profile [1]` → `error: custom op 'ninja.profile' invalid kind of attribute specified` |

MLIR keeps the keys in alphabetical order: `{target = ..., days = ...}` prints as
`{days = ..., target = ...}`.

---

## A8. Tensor data

### `ElementsAttr`: the contents of a tensor

```tablegen
def Ninja_MapOp : Ninja_Op<"map"> {
  let arguments = (ins ElementsAttr:$tiles);
  let assemblyFormat = "$tiles attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.map dense<[1, 2, 3]> : tensor<3xi32>` |
| Or | `ninja.map dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf32>` |
| C++ getter | `::mlir::ElementsAttr getTiles();` |
| Wrong | `ninja.map [1, 2]` → `error: custom op 'ninja.map' invalid kind of attribute specified` |

`dense<...>` holds the values; `: tensor<...>` says their shape and type. This is what
`arith.constant dense<...>` uses.

---

## A9. Names: defining and referring

### `SymbolNameAttr`: a name an op **defines**

```tablegen
def Ninja_TechniqueOp : Ninja_Op<"technique"> {
  let arguments = (ins SymbolNameAttr:$name);
  let assemblyFormat = "$name attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.technique @fireball` |
| Stored | `<{name = "fireball"}>` (stored as plain text) |
| C++ getter | `::llvm::StringRef getName();` |
| Wrong | `ninja.technique "fireball"` → `error: custom op 'ninja.technique' expected valid '@'-identifier for symbol name` |

### `FlatSymbolRefAttr`: a name an op **refers to**

```tablegen
def Ninja_UseOp : Ninja_Op<"use"> {
  let arguments = (ins FlatSymbolRefAttr:$technique);
  let assemblyFormat = "$technique attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.use @fireball` |
| Stored | `<{technique = @fireball}>` (stored as a reference) |
| C++ getter | `::llvm::StringRef getTechnique();` |
| Wrong | `ninja.use fireball` → `error: expected attribute value` (the `@` is required) |

Like `func.func @main` (defines) and `func.call @main` (refers). To make MLIR check that the name
really exists, see `Symbol` and `SymbolTable` in chapter 7.

---

## A10. Any attribute, and attributes that carry a type

### `AnyAttr`: anything at all

```tablegen
def Ninja_NoteOp : Ninja_Op<"note"> {
  let arguments = (ins AnyAttr:$text);
  let assemblyFormat = "$text attr-dict";
}
```

| You write | Printed back |
|---|---|
| `ninja.note 3` | `ninja.note 3 : i64` (a bare number becomes `i64`) |
| `ninja.note "x"` | `ninja.note "x"` |
| `ninja.note [1, 2]` | `ninja.note [1, 2]` |

C++ getter: `::mlir::Attribute getText();`. Use it rarely: it checks nothing.

### `TypedAttrInterface`: any attribute that has a type

```tablegen
def Ninja_ValueOp : Ninja_Op<"value"> {
  let arguments = (ins TypedAttrInterface:$v);
  let assemblyFormat = "$v attr-dict";
}
```

Needs `include "mlir/IR/BuiltinAttributeInterfaces.td"`.

| You write | Result |
|---|---|
| `ninja.value 3 : i32` | ok |
| `ninja.value 2.5 : f16` | ok, prints `2.500000e+00 : f16` |
| `ninja.value 3` | ok, `3 : i64` (the default type) |
| `ninja.value "x"` | ok: a string counts as typed (its type is `none`) |
| `ninja.value [1, 2]` | `error: custom op 'ninja.value' invalid kind of attribute specified` |

C++ getter: `::mlir::TypedAttr getV();`, and `getV().getType()` gives its type.

---

## A11. Wrappers: optional, default, rules

### `OptionalAttr<...>`: may be left out

```tablegen
def Ninja_ClimbOp : Ninja_Op<"climb"> {
  let arguments = (ins I32Attr:$height, OptionalAttr<I32Attr>:$rope);
  let assemblyFormat = "$height (`rope` $rope^)? attr-dict";
}
```

| | |
|---|---|
| You write | `ninja.climb 12` or `ninja.climb 12 rope 3` |
| Stored | `<{height = 12 : i32}>` or `<{height = 12 : i32, rope = 3 : i32}>` |
| C++ getters | `uint32_t getHeight();` and `::std::optional<uint32_t> getRope();` |
| Wrong | `ninja.climb rope 3` → `error: expected attribute value` (height is required) |

In C++: `if (auto r = op.getRope()) use(*r);`

### `DefaultValuedAttr<..., "value">`: has a default

```tablegen
def Ninja_ThrowOp : Ninja_Op<"throw"> {
  let arguments = (ins DefaultValuedAttr<I32Attr, "1">:$count);
  let assemblyFormat = "(`count` $count^)? attr-dict";
}
```

| You write | Printed back | Stored |
|---|---|---|
| `ninja.throw` | `ninja.throw` | `<{count = 1 : i32}>` |
| `ninja.throw count 5` | `ninja.throw count 5` | `<{count = 5 : i32}>` |
| `ninja.throw count 1` | `ninja.throw` | `<{count = 1 : i32}>` (default is hidden) |

C++ getter: `uint32_t getCount();`, never missing.

### `ConfinedAttr<..., [rules]>`: extra rules on the value

```tablegen
def Ninja_HealOp : Ninja_Op<"heal"> {
  let arguments = (ins ConfinedAttr<I32Attr, [IntMinValue<1>, IntMaxValue<100>]>:$points);
  let assemblyFormat = "$points attr-dict";
}
```

| You write | Result |
|---|---|
| `ninja.heal 50` | ok |
| `ninja.heal 0` or `ninja.heal 101` | `error: 'ninja.heal' op attribute 'points' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 1 whose maximum value is 100` |

The full list of rules is in chapter 4, section 4.5.

---

## Summary table

| Kind | Written as | C++ getter |
|---|---|---|
| `I32Attr` | `3` | `uint32_t` |
| `I64Attr` | `5000` | `uint64_t` |
| `SI32Attr` | `-2` | `int32_t` |
| `UI32Attr` | `7` | `uint32_t` |
| `F32Attr` | `0.5` | `llvm::APFloat` |
| `BoolAttr` | `true` | `bool` |
| `UnitAttr` | `quiet` (or nothing) | `bool` |
| `StrAttr` | `"hello"` | `llvm::StringRef` |
| `TypeAttr` | `f32` | `mlir::Type` |
| `ArrayAttr` | `[1, "rope"]` | `mlir::ArrayAttr` |
| `StrArrayAttr` | `["kenji", "aiko"]` | `mlir::ArrayAttr` |
| `DenseI64ArrayAttr` | `[1, 2, 3]` | `llvm::ArrayRef<int64_t>` |
| `DictionaryAttr` | `{age = 20 : i32}` | `mlir::DictionaryAttr` |
| `ElementsAttr` | `dense<[1, 2]> : tensor<2xi32>` | `mlir::ElementsAttr` |
| `SymbolNameAttr` | `@fireball` | `llvm::StringRef` |
| `FlatSymbolRefAttr` | `@fireball` | `llvm::StringRef` |
| `AnyAttr` | anything | `mlir::Attribute` |
| `TypedAttrInterface` | `3 : i32` | `mlir::TypedAttr` |
| `OptionalAttr<I32Attr>` | `rope 3` (or nothing) | `std::optional<uint32_t>` |
| `DefaultValuedAttr<I32Attr, "1">` | `count 5` (or nothing) | `uint32_t` |
| `ConfinedAttr<I32Attr, [...]>` | `50` | `uint32_t` |

Every attribute also has a second getter ending in `Attr` (`getPowerAttr()`), which returns the
attribute object instead of the plain value.

---

# Part B: Drills

Write each op yourself. For each drill:

- **Must accept** lines should print back without errors.
- **Must reject** lines should give an error.

Then open the answer and compare. Small differences in names are fine, as long as the IR works.

Tick them off as you go:

| # | Attribute | Done |
|---|---|---|
| B1 | `I32Attr` | ☐ |
| B2 | `I64Attr` | ☐ |
| B3 | `SI32Attr` | ☐ |
| B4 | `UI32Attr` | ☐ |
| B5 | `F32Attr` | ☐ |
| B6 | `BoolAttr` | ☐ |
| B7 | `StrAttr` | ☐ |
| B8 | `UnitAttr` | ☐ |
| B9 | `TypeAttr` | ☐ |
| B10 | `ArrayAttr` | ☐ |
| B11 | `StrArrayAttr` | ☐ |
| B12 | `DenseI64ArrayAttr` | ☐ |
| B13 | `DictionaryAttr` | ☐ |
| B14 | `ElementsAttr` | ☐ |
| B15 | `SymbolNameAttr` | ☐ |
| B16 | `FlatSymbolRefAttr` | ☐ |
| B17 | `AnyAttr` | ☐ |
| B18 | `TypedAttrInterface` | ☐ |
| B19 | `OptionalAttr` | ☐ |
| B20 | `DefaultValuedAttr` | ☐ |
| B21 | `ConfinedAttr` | ☐ |
| B22 | all together, plus C++ | ☐ |

---

### B1. `ninja.punch` (`I32Attr`)

A punch with a `force`.

- Must accept: `ninja.punch 4`

<details><summary>Answer</summary>

```tablegen
def Ninja_PunchOp : Ninja_Op<"punch"> {
  let arguments = (ins I32Attr:$force);
  let assemblyFormat = "$force attr-dict";
}
```
</details>

### B2. `ninja.wait` (`I64Attr`)

Wait a number of `millis` (milliseconds), which can be large.

- Must accept: `ninja.wait 30000`

<details><summary>Answer</summary>

```tablegen
def Ninja_WaitOp : Ninja_Op<"wait"> {
  let arguments = (ins I64Attr:$millis);
  let assemblyFormat = "$millis attr-dict";
}
```
</details>

### B3. `ninja.turn` (`SI32Attr`)

Turn by some `degrees`; left is negative. The C++ getter must return a signed number.

- Must accept: `ninja.turn -90`
- Check: in `NinjaOps.h.inc`, `getDegrees()` returns `int32_t`.

<details><summary>Answer</summary>

```tablegen
def Ninja_TurnOp : Ninja_Op<"turn"> {
  let arguments = (ins SI32Attr:$degrees);
  let assemblyFormat = "$degrees attr-dict";
}
```
</details>

### B4. `ninja.arrows` (`UI32Attr`)

A number of arrows; never negative.

- Must accept: `ninja.arrows 12`
- Must reject: `ninja.arrows -1`

<details><summary>Answer</summary>

```tablegen
def Ninja_ArrowsOp : Ninja_Op<"arrows"> {
  let arguments = (ins UI32Attr:$amount);
  let assemblyFormat = "$amount attr-dict";
}
```

`ninja.arrows -1` → `error: negative integer literal not valid for unsigned integer type`
</details>

### B5. `ninja.balance` (`F32Attr`)

A balance `level` between 0 and 1, as a 32-bit float.

- Must accept: `ninja.balance 0.75` (prints as `7.500000e-01`)

<details><summary>Answer</summary>

```tablegen
def Ninja_BalanceOp : Ninja_Op<"balance"> {
  let arguments = (ins F32Attr:$level);
  let assemblyFormat = "$level attr-dict";
}
```
</details>

### B6. `ninja.disguise` (`BoolAttr`)

Turn a disguise on or off.

- Must accept: `ninja.disguise false`, `ninja.disguise true`

<details><summary>Answer</summary>

```tablegen
def Ninja_DisguiseOp : Ninja_Op<"disguise"> {
  let arguments = (ins BoolAttr:$on);
  let assemblyFormat = "$on attr-dict";
}
```
</details>

### B7. `ninja.whisper` (`StrAttr`)

Whisper a `message`.

- Must accept: `ninja.whisper "go"`

<details><summary>Answer</summary>

```tablegen
def Ninja_WhisperOp : Ninja_Op<"whisper"> {
  let arguments = (ins StrAttr:$message);
  let assemblyFormat = "$message attr-dict";
}
```
</details>

### B8. `ninja.vanish` (`UnitAttr`)

Vanish, optionally `silent`.

- Must accept: `ninja.vanish` and `ninja.vanish silent`

<details><summary>Answer</summary>

```tablegen
def Ninja_VanishOp : Ninja_Op<"vanish"> {
  let arguments = (ins UnitAttr:$silent);
  let assemblyFormat = "(`silent` $silent^)? attr-dict";
}
```
</details>

### B9. `ninja.convert_to` (`TypeAttr`)

Name the `target` type to convert to.

- Must accept: `ninja.convert_to i64`

<details><summary>Answer</summary>

```tablegen
def Ninja_ConvertToOp : Ninja_Op<"convert_to"> {
  let arguments = (ins TypeAttr:$target);
  let assemblyFormat = "$target attr-dict";
}
```
</details>

### B10. `ninja.pack` (`ArrayAttr`)

Pack a list of mixed `items`.

- Must accept: `ninja.pack [3, "kunai", true]`

<details><summary>Answer</summary>

```tablegen
def Ninja_PackOp : Ninja_Op<"pack"> {
  let arguments = (ins ArrayAttr:$items);
  let assemblyFormat = "$items attr-dict";
}
```
</details>

### B11. `ninja.allies` (`StrArrayAttr`)

A list of ally `names`; only strings allowed.

- Must accept: `ninja.allies ["kenji", "aiko"]`
- Must reject: `ninja.allies [1]`

<details><summary>Answer</summary>

```tablegen
def Ninja_AlliesOp : Ninja_Op<"allies"> {
  let arguments = (ins StrArrayAttr:$names);
  let assemblyFormat = "$names attr-dict";
}
```

`ninja.allies [1]` → `error: 'ninja.allies' op attribute 'names' failed to satisfy constraint: string array attribute`
</details>

### B12. `ninja.route` (`DenseI64ArrayAttr`)

A list of `stops` (numbers). In C++ you want plain `int64_t`s.

- Must accept: `ninja.route [4, 8, 15]`

<details><summary>Answer</summary>

```tablegen
def Ninja_RouteOp : Ninja_Op<"route"> {
  let arguments = (ins DenseI64ArrayAttr:$stops);
  let assemblyFormat = "$stops attr-dict";
}
```
</details>

### B13. `ninja.mission` (`DictionaryAttr`)

Mission `details` as named values.

- Must accept: `ninja.mission {target = "castle", days = 3 : i32}`
- Question: how does it print back, and why?

<details><summary>Answer</summary>

```tablegen
def Ninja_MissionOp : Ninja_Op<"mission"> {
  let arguments = (ins DictionaryAttr:$details);
  let assemblyFormat = "$details attr-dict";
}
```

It prints as `ninja.mission {days = 3 : i32, target = "castle"}`: MLIR keeps dictionary keys in
alphabetical order.
</details>

### B14. `ninja.grid` (`ElementsAttr`)

A 2x2 grid of `cells`, stored as tensor data.

- Must accept: `ninja.grid dense<[[1, 0], [0, 1]]> : tensor<2x2xi8>`

<details><summary>Answer</summary>

```tablegen
def Ninja_GridOp : Ninja_Op<"grid"> {
  let arguments = (ins ElementsAttr:$cells);
  let assemblyFormat = "$cells attr-dict";
}
```
</details>

### B15. `ninja.jutsu` (`SymbolNameAttr`)

**Define** a technique with a `name`.

- Must accept: `ninja.jutsu @shadow_clone`

<details><summary>Answer</summary>

```tablegen
def Ninja_JutsuOp : Ninja_Op<"jutsu"> {
  let arguments = (ins SymbolNameAttr:$name);
  let assemblyFormat = "$name attr-dict";
}
```
</details>

### B16. `ninja.perform` (`FlatSymbolRefAttr`)

**Refer to** a technique by name.

- Must accept: `ninja.perform @shadow_clone`
- Question: in the generic form, how is it stored, compared with B15?

<details><summary>Answer</summary>

```tablegen
def Ninja_PerformOp : Ninja_Op<"perform"> {
  let arguments = (ins FlatSymbolRefAttr:$jutsu);
  let assemblyFormat = "$jutsu attr-dict";
}
```

B15 stores `name = "shadow_clone"` (text); B16 stores `jutsu = @shadow_clone` (a reference).
</details>

### B17. `ninja.tag` (`AnyAttr`)

A `label` of any kind.

- Must accept: `ninja.tag 42` and `ninja.tag "x"`
- Question: why does `ninja.tag 42` print as `ninja.tag 42 : i64`?

<details><summary>Answer</summary>

```tablegen
def Ninja_TagOp : Ninja_Op<"tag"> {
  let arguments = (ins AnyAttr:$label);
  let assemblyFormat = "$label attr-dict";
}
```

`AnyAttr` does not say which type the number has, so MLIR uses the default integer type, `i64`,
and prints it so the text can be read back exactly.
</details>

### B18. `ninja.seal` (`TypedAttrInterface`)

A `code` that carries its own type.

- Must accept: `ninja.seal 7 : i16`

<details><summary>Answer</summary>

```tablegen
def Ninja_SealOp : Ninja_Op<"seal"> {
  let arguments = (ins TypedAttrInterface:$code);
  let assemblyFormat = "$code attr-dict";
}
```

Remember `include "mlir/IR/BuiltinAttributeInterfaces.td"`.
</details>

### B19. `ninja.leap` (`OptionalAttr`)

A leap of some `distance`, with an optional number of `spins`.

- Must accept: `ninja.leap 5` and `ninja.leap 5 spin 2`

<details><summary>Answer</summary>

```tablegen
def Ninja_LeapOp : Ninja_Op<"leap"> {
  let arguments = (ins I32Attr:$distance, OptionalAttr<I32Attr>:$spins);
  let assemblyFormat = "$distance (`spin` $spins^)? attr-dict";
}
```
</details>

### B20. `ninja.bow` (`DefaultValuedAttr`)

Bow a number of `times`; the default is 1.

- Must accept: `ninja.bow`, `ninja.bow times 3`
- Question: what does `ninja.bow times 1` print back as?

<details><summary>Answer</summary>

```tablegen
def Ninja_BowOp : Ninja_Op<"bow"> {
  let arguments = (ins DefaultValuedAttr<I32Attr, "1">:$times);
  let assemblyFormat = "(`times` $times^)? attr-dict";
}
```

`ninja.bow times 1` prints as `ninja.bow`: the default value is not shown.
</details>

### B21. `ninja.focus` (`ConfinedAttr`)

A focus `level` from 1 to 10, checked without C++.

- Must accept: `ninja.focus 10`
- Must reject: `ninja.focus 0`, `ninja.focus 11`

<details><summary>Answer</summary>

```tablegen
def Ninja_FocusOp : Ninja_Op<"focus"> {
  let arguments = (ins ConfinedAttr<I32Attr, [IntMinValue<1>, IntMaxValue<10>]>:$level);
  let assemblyFormat = "$level attr-dict";
}
```

`error: 'ninja.focus' op attribute 'level' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 1 whose maximum value is 10`
</details>

---

### B22. Final drill: `ninja.ambush` (several attributes, plus C++)

An ambush has:

| Part | Kind | Rule |
|---|---|---|
| `target` | text | required |
| `position` | a list of exactly **2** numbers | checked in `.td` (hint: `DenseArrayCount<2>`) |
| `allies` | a list of strings | required |
| `silent` | a flag | optional |
| `delay` | a 64-bit number, written after `wait` | optional |

Plus one rule in **C++**: neither position number may be negative. Use the getter
`getPosition()` in a `verify()`.

- Must accept:
  - `ninja.ambush "castle" at [3, 7] with ["kenji", "aiko"] silent wait 500`
  - `ninja.ambush "castle" at [3, 7] with ["kenji"]`
- Must reject:
  - `ninja.ambush "castle" at [3] with ["kenji"]` (only one number)
  - `ninja.ambush "castle" at [3, -1] with ["kenji"]` (negative)

<details><summary>Answer</summary>

```tablegen
def Ninja_AmbushOp : Ninja_Op<"ambush"> {
  let arguments = (ins
    StrAttr:$target,
    ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<2>]>:$position,
    StrArrayAttr:$allies,
    UnitAttr:$silent,
    OptionalAttr<I64Attr>:$delay
  );
  let assemblyFormat = [{
    $target `at` $position `with` $allies (`silent` $silent^)? (`wait` $delay^)? attr-dict
  }];
  let hasVerifier = 1;
}
```

In `lib/NinjaDialect.cpp`:

```cpp
// AmbushOp: both position numbers must be zero or more.
LogicalResult AmbushOp::verify() {
  for (int64_t p : getPosition())     // getPosition() is ArrayRef<int64_t>
    if (p < 0)
      return emitOpError("position must not be negative, got ") << p;
  return success();
}
```

Results:

```
ninja.ambush "castle" at [3] with ["kenji"]
error: 'ninja.ambush' op attribute 'position' failed to satisfy constraint: i64 dense array attribute with exactly 2 elements

ninja.ambush "castle" at [3, -1] with ["kenji"]
error: 'ninja.ambush' op position must not be negative, got -1
```

Generic form of the first accepted line:

```mlir
"ninja.ambush"() <{allies = ["kenji", "aiko"], delay = 500 : i64, position = array<i64: 3, 7>, silent, target = "castle"}> : () -> ()
```
</details>

---

## When you finish

You have now written every common kind of attribute by hand. Two good next steps:

- Add a test in `test/` for two or three of your drills (one valid line, one invalid line each).
- Go back to [chapter 4](04-attributes.md) exercise 4.2 and fix the real `ninja.strike` with
  `ConfinedAttr`.
