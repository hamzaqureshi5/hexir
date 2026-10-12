# 8. Interface traits

## In one minute

**What.** An interface is a set of **questions a pass can ask any op**, where each op gives its own
answer: "Do you read or write memory?" "Is it safe to run you early?" "What is your result type?"
Interfaces go in the same `[ ... ]` list as traits, which is why this guide calls them interface
traits. `Pure` is the most common one.

**Why.** General passes such as CSE (merge duplicates), dead-code removal and loop-invariant code
motion must work on ops they have never seen, from dialects that did not exist when the pass was
written. A trait can only say yes or no; an interface lets each op give a detailed answer, for
example "I only read operand 0". Without these answers, a pass must assume the worst and leave
the op alone.

**How.** An interface is a C++ class with methods. Your op either gets a ready-made answer from
the list (`NoMemoryEffect`, `Pure`), or you write the answer yourself in C++ after asking ODS to
declare it (`DeclareOpInterfaceMethods<...>`). Passes then call the interface's methods on any op.

**Where it fits.** Mostly inside passes; some interfaces also help the parser (type inference) or
the printer (result names):
`file.mlir → PARSE → VERIFY → [PASSES] → PRINT → output`

**After this chapter** you can tell passes what your op does to memory, whether it is safe to move,
how to compute its result type, and how to name its results.

---

## 8.1 Traits vs. interfaces

**The idea.**

- A **trait** is a fixed promise. The op either has it or not. (`Commutative`)
- An **interface** is a set of **questions** a pass can ask the op, like "do you write to memory?"
  or "what is your result type?". Each op can give its own answer.

**Everyday picture.** A trait is a sticker: "fragile". An interface is a form the package can fill
in: "weight: ___, contents: ___".

**Both go in the same trait list.** There are three ways an interface appears there:

| Written as | Meaning |
|---|---|
| `NoMemoryEffect`, `Pure`, ... | a ready-made answer; nothing to write in C++ |
| `DeclareOpInterfaceMethods<SomeInterface>` | ODS declares the methods, **you write them** in C++ |
| `DeclareOpInterfaceMethods<SomeInterface, ["methodName"]>` | the same, also declaring an optional method |

The interface's `.td` file must be included, and its C++ header must be in `NinjaDialect.h`.

---

## 8.2 Memory effects

Passes need to know whether an op reads or writes memory before they remove, merge or move it.
Interfaces in `mlir/Interfaces/SideEffectInterfaces.td`, header
`mlir/Interfaces/SideEffectInterfaces.h`.

| Written as | Means |
|---|---|
| `NoMemoryEffect` | does not touch memory at all |
| `MemoryEffects<[MemRead]>` | reads memory |
| `MemoryEffects<[MemWrite]>` | writes memory |
| `MemoryEffects<[MemAlloc]>`, `MemoryEffects<[MemFree]>` | allocates / frees memory |
| `Arg<AnyMemRef, "doc", [MemRead]>:$buf` | reads **this operand** (better than a whole-op effect) |
| `Res<AnyMemRef, "doc", [MemAlloc]>:$buf` | allocates **this result** |
| `RecursiveMemoryEffects` | my effects are the effects of the ops inside my regions |
| nothing | **unknown**: passes assume the worst |

### Effect on CSE (merging identical ops)

```mlir
func.func @cse(%x: f64) -> (...) {
  %p1 = ninja.pure_op %x      // Pure
  %p2 = ninja.pure_op %x
  %n1 = ninja.nomem %x        // NoMemoryEffect
  %n2 = ninja.nomem %x
  %w1 = ninja.writes %x       // MemoryEffects<[MemWrite]>
  %w2 = ninja.writes %x
  %i1 = ninja.noinfo %x       // no effect information
  %i2 = ninja.noinfo %x
  return ...
}
```

After `--cse` (real output):

```mlir
%0 = ninja.pure_op %arg0     // the two pure_ops merged into one
%1 = ninja.nomem %arg0       // merged
%2 = ninja.writes %arg0      // both kept: writing twice is not the same as writing once
%3 = ninja.writes %arg0
%4 = ninja.noinfo %arg0      // both kept: unknown, so assumed unsafe
%5 = ninja.noinfo %arg0
```

### Effect on dead-code removal

```mlir
func.func @dce(%x: f64) {
  %p = ninja.pure_op %x       // results never used
  %n = ninja.nomem %x
  %w = ninja.writes %x
  %i = ninja.noinfo %x
  return
}
```

After `--canonicalize`:

```mlir
%0 = ninja.writes %arg0      // kept: the write itself matters
%1 = ninja.noinfo %arg0      // kept: unknown
```

`pure_op` and `nomem` were removed because nothing used their results and they have no other
effect.

### Effects on specific operands

```tablegen
def Ninja_ReadsOp : Ninja_Op<"reads"> {
  let arguments = (ins Arg<AnyMemRef, "the buffer", [MemRead]>:$buf);
  let results = (outs F64:$r);
}
def Ninja_StoresOp : Ninja_Op<"stores"> {
  let arguments = (ins F64:$value, Arg<AnyMemRef, "the buffer", [MemWrite]>:$buf);
}
```

```mlir
%a = ninja.reads %m : memref<4xf64>
%b = ninja.reads %m : memref<4xf64>       // same read, nothing written in between
ninja.stores %v, %m : memref<4xf64>
%c = ninja.reads %m : memref<4xf64>       // a write happened in between
```

After `--cse`: `%b` is merged into `%a`; `%c` is kept, because the store may have changed the
buffer.

### `RecursiveMemoryEffects`

For ops with regions: "my effects are whatever the ops inside me do".

```mlir
ninja.box { %p = ninja.pure_op %x }        // RecursiveMemoryEffects, pure inside
ninja.box { %w = ninja.writes %x }         // RecursiveMemoryEffects, writes inside
ninja.plainbox { %p = ninja.pure_op %x }   // no effect information
```

After `--canonicalize`:

```mlir
ninja.box {
  %0 = ninja.writes %arg0
}
ninja.plainbox {
}
```

The first `box` is gone entirely: nothing inside has an effect. The second is kept. `plainbox`
stays even though it is now empty, because without effect information MLIR must assume it does
something.

---

## 8.3 Speculation: "is it safe to run early?"

**The idea.** Some optimizations move an op to a place where it may run **even when the original
code would not have run it**, for example moving it out of a loop that might run zero times. That
is only safe if running the op "for nothing" cannot crash or misbehave. An op that divides by an
input is **not** safe: the input might be zero on a path that would never have reached it.

| Written as | Means |
|---|---|
| `AlwaysSpeculatable` | always safe to run early |
| `RecursivelySpeculatable` | safe if everything inside my regions is |
| `DeclareOpInterfaceMethods<ConditionallySpeculatable>` | you write `getSpeculatability()` to decide |
| nothing | not speculatable |

### `Pure` = `NoMemoryEffect` + `AlwaysSpeculatable`

This is the exact definition in `SideEffectInterfaces.td`:

```tablegen
def Pure : TraitList<[AlwaysSpeculatable, NoMemoryEffect]>;
```

**What you see.** `--loop-invariant-code-motion` moves ops out of a loop only if they are
memory-effect free **and** speculatable:

```mlir
scf.for %i = %c0 to %n step %c1 {
  %p = ninja.pure_op %x       // Pure
  %m = ninja.nomem %x         // NoMemoryEffect only
  %s = ninja.spec %x          // NoMemoryEffect + AlwaysSpeculatable
  "test.use"(%p, %m, %s) : (f64, f64, f64) -> ()
}
```

After the pass (real output):

```mlir
%0 = ninja.pure_op %arg0      // moved out
%1 = ninja.spec %arg0         // moved out
scf.for %arg2 = %c0 to %arg1 step %c1 {
  %2 = ninja.nomem %arg0      // stays: not known to be safe to run early
  "test.use"(%0, %2, %1) : (f64, f64, f64) -> ()
}
```

So `Pure` is not just "no side effects". It also says "safe to run early". If your op can trap
(divide by zero, out-of-bounds access), use `NoMemoryEffect`, not `Pure`.

---

## 8.4 Type inference: `InferTypeOpInterface`

**The idea.** The op computes its own result types from its operands and attributes, so nobody has
to write them: not the parser, not the builders.

**Ready-made forms.** `SameOperandsAndResultType` (chapter 7) and an exact result constraint like
`F64` already provide it. ODS adds `InferTypeOpInterface::Trait` automatically in those cases;
you can see it in the `class StrikeOp` line in chapter 7.2.

**Your own rule: `InferTypeOpAdaptor`.** Write one C++ function, and ODS uses it everywhere.

```tablegen
def Ninja_ElemOp : Ninja_Op<"elem", [Pure, InferTypeOpAdaptor]> {
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";     // no result type
}
```

```cpp
// The result type is the element type of the input tensor.
LogicalResult ElemOp::inferReturnTypes(MLIRContext *, std::optional<Location>,
                                       ElemOp::Adaptor adaptor,
                                       SmallVectorImpl<Type> &inferred) {
  inferred.push_back(cast<ShapedType>(adaptor.getInput().getType()).getElementType());
  return success();
}
```

**What you see:**

```mlir
%e = ninja.elem %t : tensor<4xf32>              // result is f32, worked out for you
"ninja.elem"(%t) : (tensor<4xf32>) -> i32
// error: 'ninja.elem' op inferred type(s) 'f32' are incompatible with return type(s) of operation 'i32'
```

The interface also checks: if someone writes a different result type, it is rejected.

---

## 8.5 Nicer result names: `OpAsmOpInterface`

**The idea.** Choose the name results get when printed, instead of `%0`, `%1`.

```tablegen
def Ninja_NamedOp : Ninja_Op<"named", [Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let arguments = (ins F64:$lhs, F64:$rhs);
  let results = (outs F64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}
```

```cpp
void NamedOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "sum");
}
```

```mlir
%sum = ninja.named %arg0, %arg1
%sum_0 = ninja.named %arg0, %arg1        // MLIR adds _0, _1 to keep names unique
```

`getAsmResultNames` is an **optional** method of `OpAsmOpInterface`, which is why it is named in
the list: `DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>`. Include
`mlir/IR/OpAsmInterface.td`.

---

## 8.6 Casts: `CastOpInterface`

**The idea.** Marks an op as a type conversion, and lets you say which conversions are allowed.

```tablegen
def Ninja_CastOp : Ninja_Op<"cast", [Pure, DeclareOpInterfaceMethods<CastOpInterface>]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
}
```

```cpp
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return inputs.size() == 1 && outputs.size() == 1 &&
         isa<FloatType>(inputs.front()) && isa<FloatType>(outputs.front());
}
```

```mlir
%r = ninja.cast %a : f32 to f64          // ok
%r = ninja.cast %a : i32 to f64
// error: 'ninja.cast' op operand type 'i32' and result type 'f64' are cast incompatible
```

Include `mlir/Interfaces/CastInterfaces.td`, header `mlir/Interfaces/CastInterfaces.h`.

---

## 8.7 Other interfaces you will meet

These are central to MLIR's own dialects. They need more C++ than fits here, so this section says
what each one is for and where to see a real example in `/home/user/llvm-project/mlir/`.

| Interface | What it says | Real example |
|---|---|---|
| `ReturnLike` | "I am a terminator that returns values to my parent". A bundle that also adds `RegionBranchTerminatorOpInterface` | `scf.yield`, `func.return` |
| `RegionBranchOpInterface` | "control flows into and out of my regions like this" | `scf.if`, `scf.for` |
| `RegionBranchTerminatorOpInterface` | "I pass values out of a region" | `scf.yield`, `scf.condition` |
| `BranchOpInterface` | "I jump to other blocks, passing these values" | `cf.br`, `cf.cond_br` |
| `CallOpInterface` | "I call a function" | `func.call` |
| `CallableOpInterface` | "I can be called" | `func.func` |
| `FunctionOpInterface` | "I am a function" (name, signature, body) | `func.func`, hexir's `hextir.prim_func` |
| `SymbolUserOpInterface` | "I refer to `@names`; check they exist" | `func.call` |
| `LoopLikeOpInterface` | "I am a loop" | `scf.for` |
| `DestinationStyleOpInterface` | "I write my result into an `outs` operand" | `linalg` ops |

Find their `.td` definitions in `include/mlir/Interfaces/` and their use in
`include/mlir/Dialect/SCF/IR/SCFOps.td` and `include/mlir/Dialect/Func/IR/FuncOps.td`.

---

## Watch out

- **No effect information means "assume the worst".** An op without `NoMemoryEffect`, `Pure` or
  `MemoryEffects<...>` is never removed or merged, even if it does nothing.
- **`Pure` claims "safe to run early".** Do not put it on ops that can trap.
- **Missing includes.** Each interface needs its `.td` include and its C++ header. A missing
  header shows up as C++ errors about unknown names in `NinjaOps.h.inc`.

---

## Exercises

**8.1** `ninja.strike` has `Pure`. Write a function with two identical `ninja.strike` ops and run
`--cse`. What happens? Then change `Pure` to `MemoryEffects<[MemWrite]>` and run it again.

**8.2** A `ninja.divide` op divides two floats. Should it be `Pure` or `NoMemoryEffect`? What
about integer division? Explain using section 8.3.

**8.3** Add `InferTypeOpAdaptor` to a new op `ninja.first`, whose result is the type of its first
operand (it has two `AnyType` operands). Write `inferReturnTypes`.

**8.4** Give `ninja.add`'s result the printed name `%total` using `OpAsmOpInterface`.

[Solutions](solutions.md#chapter-8)
