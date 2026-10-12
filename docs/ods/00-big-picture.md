# 0. The big picture

Read this first. It explains what happens to an op from the moment MLIR reads it to the moment it
is written out again, and which part of a `.td` file controls each step. Every later chapter
zooms into one piece of this picture.

---

## 0.1 What problem are we solving?

MLIR lets you invent your own operations, like `ninja.strike`. To be useful, every op needs a lot
of supporting code:

| MLIR must be able to... | So someone has to write... |
|---|---|
| read the op from a text file | a **parser** |
| write the op back to text | a **printer** |
| check that the op makes sense | a **verifier** |
| let C++ code read its parts | **getters** like `getTimes()` |
| let C++ code create a new one | **builders** |
| simplify it | **folders** and **rewrite patterns** |
| tell passes what kind of op it is | **traits** and **interfaces** |

Written by hand in C++, that is a lot of code. Your ninja has 3 ops, described in about 100 lines
of `.td`. From them, ODS generates **723 lines** of C++ declarations (`NinjaOps.h.inc`) and
**1320 lines** of C++ code (`NinjaOps.cpp.inc`).

**ODS** (Operation Definition Specification) is how you avoid writing that by hand: you
**describe** each op in a `.td` file, and the `mlir-tblgen` tool **writes the C++ for you**.

---

## 0.2 Two moments: build time and run time

There are two separate moments, and it helps to keep them apart:

```
BUILD TIME (when you run cmake --build)
───────────────────────────────────────
 NinjaOps.td ──mlir-tblgen──► NinjaOps.h.inc ─┐
                              NinjaOps.cpp.inc ├─► compiled with NinjaDialect.cpp ──► ninja-opt
                                               ┘

RUN TIME (when you run ./build/ninja-opt file.mlir)
───────────────────────────────────────────────────
 file.mlir ──► parse ──► verify ──► passes ──► verify ──► print ──► output text
```

- At **build time**, your `.td` becomes C++ and gets compiled into `ninja-opt`.
- At **run time**, `ninja-opt` uses that compiled code to read, check, change and print IR.

A `.td` file never runs by itself. Everything it describes happens through the C++ it generates.

---

## 0.3 The life of one op, step by step

Take this small program (`journey.mlir`):

```mlir
func.func @train(%x: f64) -> f64 {
  %y = ninja.strike %x : f64 times 3 : f64
  %unused = ninja.add %x, %x : f64
  return %y : f64
}
```

### Step 1: Parse (text → op in memory)

MLIR reads the text and builds an **op object** in memory for each line. The **parser** for
`ninja.strike` was generated from its `assemblyFormat`.

In memory, the op is not text any more. It is a structure with a name, operands, attributes and
result types. The **generic form** shows that structure directly (real output):

```mlir
%0 = "ninja.strike"(%arg0) <{times = 3 : i32}> : (f64) -> f64
//    └── name ──┘ └operands┘ └─attributes─┘    └─ types ─┘
```

If the text has the wrong **shape**, parsing stops right here:

```mlir
%y = ninja.strike %x : f64 3 : f64        // forgot "times"
// error: custom op 'ninja.strike' expected 'times'
```

*Controlled by:* `assemblyFormat` (chapter 5), plus the operands, attributes and types it
mentions (chapters 3 and 4).

### Step 2: Verify (does the op make sense?)

The text had the right shape, but are the values allowed? MLIR runs the op's **verifier** right
after parsing:

```mlir
%y = ninja.strike %x : f64 times 0 : f64
// error: 'ninja.strike' op times must be positive, got 0
```

That rule comes from `StrikeOp::verify()` in `lib/NinjaDialect.cpp`. Other rules, like "`target`
must be `f64`", come from the `.td` file with no C++ at all.

*Controlled by:* type constraints (chapter 3), attribute constraints (chapter 4), traits
(chapter 7) and `hasVerifier` (chapter 6).

### Step 3: Run passes (change the IR)

A **pass** walks over the IR and changes it. `--canonicalize` is a pass that simplifies ops,
using rules each op provides. Real output:

```mlir
// ./build/ninja-opt journey.mlir --canonicalize
func.func @train(%arg0: f64) -> f64 {
  %0 = ninja.strike %arg0 : f64 times 3 : f64
  return %0 : f64
}
```

The unused `ninja.add` is gone. The pass knew it could delete it because `ninja.add` has the
`Pure` trait, which says "I have no side effects". The pass never needed to know what `ninja.add`
does; it only asked about its traits.

*Controlled by:* traits and interfaces (chapters 7 and 8), folders and rewrite patterns
(chapters 10 and 13), builders for creating new ops (chapter 9), and passes (chapter 14).

### Step 4: Verify again

After **every** pass, MLIR verifies the IR again (`--verify-each` is on by default). If a pass
creates a broken op, you get an error pointing at that pass, instead of a mysterious crash later.

### Step 5: Print (op in memory → text)

Finally, MLIR writes the IR back out. The **printer** for `ninja.strike` was generated from the
same `assemblyFormat` as the parser, so whatever it prints can be read back in (a **round trip**).

*Controlled by:* `assemblyFormat` (chapter 5).

---

## 0.4 The whole picture in one diagram

```
                 ┌──────────────────────── your .td file describes ───────────────────────┐
                 │                                                                        │
  file.mlir ──► PARSE ─────► VERIFY ─────► PASSES ─────────────────► VERIFY ─────► PRINT ──► output
                 │            │             │                          │            │
           assemblyFormat     │       traits & interfaces          (same as     assemblyFormat
           (ch. 5)            │       (ch. 7, 8)                    before)      (ch. 5)
                              │       folders, patterns, DRR
                     type constraints (ch. 3)   (ch. 10, 13)
                     attribute rules  (ch. 4)   builders (ch. 9)
                     traits           (ch. 7)   passes   (ch. 14)
                     hasVerifier      (ch. 6)

  Underneath everything:  TableGen language (ch. 1, 15)  ·  dialect and op structure (ch. 2)
                          regions (ch. 11)  ·  your own types/attributes/enums (ch. 12)
                          build plumbing and generated code (ch. 16)
```

---

## 0.5 What an op is made of

Every op, whatever its dialect, is built from the same few parts. Chapter 2 covers them in
detail:

| Part | Everyday picture | In `%y = ninja.strike %x : f64 times 3 : f64` |
|---|---|---|
| **Name** | the label on a box | `ninja.strike` |
| **Operands** | wires coming **in** from other boxes | `%x` |
| **Attributes** | words **printed on** the box | `times = 3` |
| **Results** | wires going **out** | `%y` |
| **Regions** | smaller boxes **inside** the box | (none here) |

In C++, MLIR stores every op as a generic `Operation`. The class ODS generates, `StrikeOp`, is a
typed **view** of that `Operation`: it adds named getters like `getTarget()` and `getTimes()`, so
you do not have to write `op->getOperand(0)`.

---

## 0.6 How to read the rest of this guide

Each chapter starts with **"In one minute"**:

- **What** the topic is
- **Why** it exists: the problem it solves
- **How** it works underneath
- **Where** it fits in the picture above

Then come the details, examples with real output, common mistakes and exercises.

Suggested order: 1 → 2 → 3 → 4 → 5 → 6 (you can now define and check ops), then 7 → 8 (traits),
then 9 → 10 → 11 (behaviour), then 12 → 16 as you need them.
