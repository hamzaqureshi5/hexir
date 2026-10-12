# 11. Regions and successors

## In one minute

**What.** A **region** is code **inside** an op: a list of blocks, each a list of ops. A function
body, a loop body and the two branches of an `if` are all regions. A **successor** is a block a
branch op jumps to.

**Why.** Many compilers only have flat lists of instructions with jumps between them. That loses
structure: a pass cannot easily see "this is a loop". MLIR keeps structure by letting ops contain
code. A loop op holds its body, so passes can work on the loop as a whole. Functions and modules
are ops with regions too.

**How.** You declare regions in the `.td` file with a size rule (`SizedRegion<1>` means exactly
one block). Traits (chapter 7) say how the region must end (with a terminator or not) and what it
may see from outside. Branch ops declare successors, and an interface tells MLIR which values they
pass to the target block.

**Where it fits.** Everywhere: every `.mlir` file is a module op whose region holds functions,
whose regions hold your ops:
`file.mlir → [PARSE] → [VERIFY] → [PASSES] → [PRINT] → output`

**After this chapter** you can define ops that contain code, control their shape, and define branch
ops.

---

## 11.1 What a region is

**The idea.** A **region** is a list of blocks; a **block** is a list of ops. An op with a region
contains code. Functions, loops and `if` all work this way.

**Everyday picture.** A box that contains smaller boxes.

```mlir
ninja.dojo {                    // the op
  %z = ninja.constant 1.0       // ops inside its region
}
```

---

## 11.2 Declaring regions

```tablegen
let regions = (region SizedRegion<1>:$body);
```

| Constraint | Means |
|---|---|
| `AnyRegion` | any number of blocks (0 or more) |
| `SizedRegion<1>` | exactly 1 block |
| `MinSizedRegion<n>` | at least n blocks |
| `MaxSizedRegion<n>` | at most n blocks |
| `VariadicRegion<AnyRegion>` | any number of regions |

Each `$name` gets a getter that returns the `Region &`: `getBody()`.

**Example** *(lab op)*: an `if`-like op with two regions.

```tablegen
def Ninja_ChoiceOp : Ninja_Op<"choice", [NoTerminator]> {
  let arguments = (ins I1:$cond);
  let regions = (region SizedRegion<1>:$then, MaxSizedRegion<1>:$otherwise);
  let assemblyFormat = "$cond $then `else` $otherwise attr-dict";
}
```

```mlir
ninja.choice %c {
  %a = ninja.neg %x : f64
} else {
  %b = ninja.abs %x : f64
}
```

`$then` must have exactly one block. Giving it none (in generic form) fails:

```
error: 'ninja.choice' op region #0 ('then') failed to verify constraint: region with 1 blocks
```

---

## 11.3 How a region ends: terminators

In a normal region, **every block must end with a terminator**: an op with the `Terminator`
trait, like `func.return`. You have three choices for your op's region:

| Choice | Traits | You write in text |
|---|---|---|
| Explicit terminator | regions end with an op that has `Terminator` | `ninja.yield` at the end |
| Implicit terminator | `SingleBlockImplicitTerminator<"YieldOp">` | nothing: MLIR adds the yield |
| No terminator | `NoTerminator` + `SingleBlock` | nothing |

**The usual pattern** *(lab ops)*: a region op and its own terminator.

```tablegen
def Ninja_DojoOp : Ninja_Op<"dojo", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def Ninja_YieldOp : Ninja_Op<"yield", [Pure, Terminator, HasParent<"DojoOp">]> {
  let assemblyFormat = "attr-dict";
}
```

```mlir
ninja.dojo {
  %z = ninja.constant 1.0
}
```

The generic form shows the yield MLIR added:

```mlir
"ninja.dojo"() ({
  %3 = "ninja.constant"() <{value = 1.000000e+00 : f64}> : () -> f64
  "ninja.yield"() : () -> ()
}) : () -> ()
```

`HasParent<"DojoOp">` stops `ninja.yield` from being used anywhere else. Note the name is the C++
class name (`DojoOp`), not the IR name.

---

## 11.4 Block arguments

The first block of a region can take **arguments**, like a function's parameters or a loop's
counter. They are written `^bb0(%i: index):`.

hexir's `hextir.for` used one for the loop counter, created in its C++ builder:

```cpp
Block &block = body->emplaceBlock();
block.addArgument(builder.getIndexType(), state.location);   // the induction variable
```

Use the trait `NoRegionArguments` (chapter 7) to forbid them.

---

## 11.5 Can the region see outside values?

By default, ops inside a region may use values defined outside it:

```mlir
func.func @f(%x: f64) {
  ninja.dojo {
    %y = ninja.neg %x : f64          // ok: %x comes from outside
  }
}
```

With `IsolatedFromAbove` (chapter 7), they may not. Functions are isolated; loops and `if` are not.

---

## 11.6 Successors: branch ops

**The idea.** A **successor** is a block that a terminator jumps to. `cf.br ^bb1` has one
successor.

```tablegen
def Ninja_JumpOp : Ninja_Op<"jump_to", [Terminator, DeclareOpInterfaceMethods<BranchOpInterface>]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = "$dest (`(` $args^ `:` type($args) `)`)? attr-dict";
}
```

```mlir
func.func @f(%x: f64) {
  ninja.jump_to ^next(%x : f64)
^next(%y: f64):
  return
}
```

| Constraint | Means |
|---|---|
| `AnySuccessor` | one block |
| `VariadicSuccessor<AnySuccessor>` | any number of blocks |

**Why `BranchOpInterface` matters.** The op passes `args` to the target block's arguments, but
MLIR only knows that if the op says so:

```cpp
SuccessorOperands JumpOp::getSuccessorOperands(unsigned index) {
  return SuccessorOperands(getArgsMutable());
}
```

With it, MLIR checks the types:

```mlir
ninja.jump_to ^next(%x : f64)
^next(%y: i32):
// error: type mismatch for bb argument #0 of successor #0
```

**Without** `BranchOpInterface`, that same mismatch is **accepted** (tested). The interface also
lets passes like dead-block removal understand the jump. Include
`mlir/Interfaces/ControlFlowInterfaces.td`, header `mlir/Interfaces/ControlFlowInterfaces.h`.

---

## Watch out

- **Region ops without a terminator choice fail.** A block that does not end with a terminator gives
  `block with no terminator`, unless the op has `NoTerminator` or an implicit terminator.
- **`HasParent<"...">` uses C++ class names.** The class name comes from the record name with the
  prefix removed: `Ninja_DojoOp` → `DojoOp`, and `L_JumpOp` → `JumpOp` even if its mnemonic is
  `jump_to`.

---

## Exercises

**11.1** Add `ninja.dojo` and `ninja.yield` from section 11.3. Check that the generic form shows
the hidden `ninja.yield`.

**11.2** Try `ninja.yield` directly inside a `func.func`. What is the error?

**11.3** Change `ninja.dojo` to `[NoTerminator, SingleBlock]` and remove `ninja.yield`. What is
different, and when would you choose each?

**11.4** Write `ninja.choice` from section 11.2 and try it with an empty `else` region.

[Solutions](solutions.md#chapter-11)
