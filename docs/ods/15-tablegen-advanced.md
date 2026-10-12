# 15. Advanced TableGen

## In one minute

**What.** TableGen features for writing **many records with little typing**: computed fields,
loops (`foreach`), groups of records (`multiclass`/`defm`), blocks that set a field on many
records (`let ... in`), checks (`assert`) and built-in functions (`!add`, `!if`, ...).

**Why.** Large dialects have hundreds of similar ops. Without these tools you would copy and paste
the same lines over and over, and fixing a mistake would mean fixing it everywhere. MLIR's own
`.td` files use these features heavily, so you need them to read real dialects.

**How.** All of it happens inside `mlir-tblgen` while it reads the file, before any generator runs.
By the time C++ is generated, every loop and template has been expanded into plain records.
`--print-records` shows the expanded result.

**Where it fits.** Build time only, like chapter 1.

**After this chapter** you can read MLIR's own dialect files and remove repetition from yours.

All examples below are plain TableGen (no MLIR). Try them with:

```bash
/home/user/llvm-project/build/bin/llvm-tblgen --print-records your.td
```

---

## 15.1 The example

```tablegen
class Weapon<string n, int d> {
  string name = n;
  int damage = d;
  bit heavy = !gt(d, 10);                        // computed: is d > 10?
  string label = !if(heavy, "HEAVY " # n, n);    // computed from another field
}

defvar base = 10;

def Sword : Weapon<"sword", base>;

foreach w = ["star", "kunai"] in
  def Throw_ # w : Weapon<!strconcat("throwing ", w), !add(base, 5)>;

multiclass Pair<int d> {
  def _light : Weapon<NAME # " light", d>;
  def _heavy : Weapon<NAME # " heavy", !mul(d, 2)>;
}
defm Staff : Pair<4>;
```

**What you see** (real output, shortened):

```
def Sword        { name = "sword";          damage = 10; heavy = 0; label = "sword"; }
def Throw_kunai  { name = "throwing kunai"; damage = 15; heavy = 1; label = "HEAVY throwing kunai"; }
def Throw_star   { name = "throwing star";  damage = 15; heavy = 1; label = "HEAVY throwing star"; }
def Staff_light  { name = "Staff light";    damage = 4;  heavy = 0; label = "Staff light"; }
def Staff_heavy  { name = "Staff heavy";    damage = 8;  heavy = 0; label = "Staff heavy"; }
```

---

## 15.2 Statements

| Statement | Meaning |
|---|---|
| `defvar name = value;` | a named constant, used only while reading the file |
| `foreach x = [...] in def ...;` | make one record per list item |
| `multiclass M<...> { def _a ...; def _b ...; }` | a recipe for several records |
| `defm Name : M<...>;` | use the recipe; makes `Name_a`, `Name_b` |
| `NAME` | inside a `multiclass`, the name given to `defm` |
| `let field = value in { def ...; def ...; }` | set a field on every record in the block |
| `assert condition, "message";` | stop with an error if the condition is false |

### `let ... in { }`

```tablegen
let damage = 99 in {
  def Bomb1 : Weapon<"bomb", 0>;
  def Bomb2 : Weapon<"bomb", 0>;
}
```

```
def Bomb1 { name = "bomb"; damage = 99; heavy = 0; label = "bomb"; }
```

**Notice:** `damage` is 99, but `heavy` is still `0`. `heavy` was computed from the template
argument `d` (which is 0), not from the field `damage`. **`let` changes one field; it does not
re-run anything computed from the template arguments.** This trips up many people.

MLIR's own `.td` files use `let ... in` often, for example to set `cppNamespace` on many records.

### `assert`

```tablegen
class Checked<int n> {
  assert !gt(n, 0), "n must be positive";
  int value = n;
}
def Ok : Checked<3>;
def Bad : Checked<0>;
```

```
error: assertion failed: n must be positive
error: assertion failed in this record
```

MLIR uses `assert` to catch misuse while TableGen runs. For example,
`InputMatchesTypes` asserts that its two lists have the same length.

---

## 15.3 `!` operators

| Operator | Result | Example → value |
|---|---|---|
| `!add(a, b)`, `!sub`, `!mul` | arithmetic | `!add(10, 5)` → `15` |
| `!gt(a, b)`, `!lt`, `!ge`, `!le` | comparison (a `bit`) | `!gt(15, 10)` → `1` |
| `!eq(a, b)`, `!ne` | equality | `!eq(Sword.name, "sword")` → `1` |
| `!if(cond, then, else)` | choose | `!if(1, "HEAVY", "")` → `"HEAVY"` |
| `!strconcat(a, b)` | join strings | `!strconcat("throwing ", "star")` |
| `a # b` | paste (strings or names) | `Throw_ # w` → `Throw_star` |
| `!size(list)` | length | `!size([1, 2, 3])` → `3` |
| `!foreach(x, list, expr)` | map over a list | `!foreach(x, [1, 2, 3], !mul(x, 2))` → `[2, 4, 6]` |
| `!listconcat(a, b)` | join lists | `!listconcat([1, 2, 3], [4])` → `[1, 2, 3, 4]` |
| `!interleave(list, sep)` | join a list into a string | `!interleave([1, 2, 3], ", ")` → `"1, 2, 3"` |

All of these were checked with `llvm-tblgen`. `Record.field` reads a field of another record:
`Sword.name`.

---

## 15.4 Reading real MLIR `.td` files

You now know enough to read MLIR's own dialects. Good ones to start with, inside
`/home/user/llvm-project/mlir/`:

| File | Why |
|---|---|
| `examples/toy/Ch2/include/toy/Ops.td` | the official tutorial, close to ninja |
| `include/mlir/Dialect/Arith/IR/ArithOps.td` | many small ops; base classes, traits, folders |
| `include/mlir/Dialect/SCF/IR/SCFOps.td` | regions done for real: `scf.for`, `scf.if` |
| `include/mlir/Dialect/Tensor/IR/TensorOps.td` | variadic operands, complex formats |
| `include/mlir/IR/OpBase.td` | where most traits are defined (chapter 7) |

`ArithOps.td` shows the pattern you will use most: a base `class` per **kind** of op, then many
short `def`s:

```tablegen
class Arith_FloatBinaryOp<string mnemonic, list<Trait> traits = []> : ...                 // line 82
class Arith_FloatBinaryOpWithRoundingMode<string mnemonic, ...> : ...                     // line 96
def Arith_AddFOp : Arith_FloatBinaryOpWithRoundingMode<"addf", [Commutative]> { ... }    // line 1092
def Arith_SubFOp : Arith_FloatBinaryOpWithRoundingMode<"subf"> { ... }                   // line 1124
```

(Line numbers are for the LLVM in `/home/user/llvm-project`.)

---

## Exercises

**15.1** Using a `class` like `Arith_FloatBinaryOp`, make `ninja.add`, `ninja.sub` and `ninja.mul`
share one base class `Ninja_BinaryOp`, so each `def` is one or two lines.

**15.2** Write a `foreach` that makes records `Belt_white`, `Belt_yellow`, `Belt_black`, each with
a `string color` field.

**15.3** In section 15.2, how would you change `Weapon` so that `let damage = 99` also makes
`heavy` true? (Hint: compute `heavy` from the field, not the template argument.)

[Solutions](solutions.md#chapter-15)
