# TO_DO

Where optimizations go in this codebase, and which ones are worth doing.

The catalogue of *what* the three big compilers do lives in `docs/optimizations/`.
This file is the *how* and the *next*: the mechanism for adding one here, and a
checklist ordered by value.

For the order to build it in — phase structure, the infrastructure to put in
place first, and milestones with exit criteria — see
[`docs/optimizations/plan.md`](docs/optimizations/plan.md). Sections 2 to 5
below are the individual items; the plan is the sequence.

---

## 0. Fix this first

**The graph-optimization slot never runs.** `compiler/Pipelines/Pipelines.cpp`
nests the cleanup pipeline on `hexir::FuncOp`:

```cpp
OpPassManager &optPM = pm.nest<mlir::hexir::FuncOp>();
optPM.addPass(createCanonicalizerPass());
optPM.addPass(mlir::hexir::createShapeInferencePass());
optPM.addPass(createCanonicalizerPass());
optPM.addPass(createCSEPass());
```

Nothing produces `hexir.func`. `compiler/Support/Builder.cpp` creates
`func::FuncOp`, every test writes `func.func`, and `grep -rl 'hexir.func'
compiler/test/` finds nothing. So that nested pass manager matches zero
operations and all four passes are skipped. Two identical `hexir.constant` ops
survive `-emit=mlir -opt` untouched, which is the proof.

- [ ] Nest on `func::FuncOp`, or run at module scope, and add a test that CSE
      actually merges two identical constants.

Every graph-level item below depends on this. All four compute ops
(`constant`, `add`, `relu`, `linear`) already carry `Pure`, so DCE and CSE come
free once the passes run at all — and `print` correctly does not carry it.

---

## 1. How to add an optimization

Four mechanisms, and picking the wrong one is the usual way to waste a weekend.

### A. A folder or canonicalization pattern — no new pass

Runs inside `canonicalize`, which is already in the pipeline. Cheapest thing
that exists. `ConstantOp` already sets `hasFolder = 1`; that is the model.

DRR, in `include/hexir/Dialect/Hexir/IR/HexirCombine.td` (every pattern in it is
currently commented out):

```tablegen
def ReluReluOptPattern : Pat<(ReluOp (ReluOp $arg)), (ReluOp $arg)>;
```

Or C++, when the rewrite has to compute something: `let hasCanonicalizer = 1;`
on the op, then `ReluOp::getCanonicalizationPatterns` in `HexirDialect.cpp`.

### B. A pass that rewrites one dialect in place

Five edits. `compiler/Dialect/Hexir/Transforms/Partition.cpp` is 74 lines and is
the smallest complete example to copy.

| File | Edit |
| --- | --- |
| `compiler/Dialect/<D>/Transforms/MyPass.cpp` | the pass |
| `compiler/Dialect/<D>/Transforms/CMakeLists.txt` | add the filename to `hexir_library()` |
| `include/hexir/Dialect/<D>/Transforms/Passes.h` | declare `createMyPass()` |
| `compiler/Pipelines/Pipelines.cpp` | `pm.addPass(createMyPass())` |
| `compiler/test/my-pass.mlir` | a lit test with its own `%s` input |

There is no glob in the build. The CMakeLists line is not optional, and a new
directory needs an `add_subdirectory` in its parent.

Insertion point matters: before `createLowerToLinalgPass()` for graph work,
between `createPartitionPass()` and `createLowerToTIRPass()` for kernel work on
the `-emit=mlir-tir` / `hxb` path.

### C. A conversion that crosses dialects

`compiler/Conversion/<A>To<B>/`, declared in `include/hexir/Conversion/Passes.h`.
For lowering, not for optimization.

### D. Memory and execution — not a pass at all

**This is the one people get wrong.** Buffer reuse, weight residency, in-place
update and launch overhead are not MLIR passes and no pipeline change reaches
them. They live in two places:

* `compiler/Serialization/ModuleSerializer.cpp` — assigns slots, emits `ALLOC` /
  `CONST` / `DISPATCH`, decides what the runtime is *told* to do.
* `runtime/src/vm/interpreter.c` and `runtime/src/hal/` — decides what actually
  happens when it is told.

If a change touches the encoding between them, it touches
`runtime/include/hexir_runtime/program.h`, which is the contract. Protocol for
that, and it is not optional:

- [ ] bump `HEXIR_MODULE_VERSION` in `module.h`
- [ ] regenerate `runtime/test/fixtures/mlp-cpu.hxb` — the fixture failing is
      the format pin working, so change it deliberately
- [ ] update `tools/hxb-dump.py`, the deliberate second implementation of the
      reader; when it and the runtime disagree, one of them is wrong

---

## 2. Graph level — the `hexir` dialect

- [ ] CSE and DCE — free, once §0 is fixed
- [ ] `relu(relu(x))`, `add(x, 0)`, identity rewrites — DRR in `HexirCombine.td`
- [ ] Const-eval `linear` / `add` / `relu` on constant operands, via
      `LinearOp::fold` and friends. Guard on element count, or a 4096² weight
      folds at compile time and the compile never finishes
- [ ] Fold a bias into the preceding weights; fold consecutive `linear`s
- [ ] **Fusion decision: tag `linear → add → relu` as one unit.** The highest
      value item in this file. Everything else here is worth a few percent;
      this is the difference between three round trips to DRAM and one
- [ ] Merge adjacent same-device ops into a region before placement — Relay's
      `MergeCompilerRegions`, in `Partition.cpp`
- [ ] Transfer insertion, so a mixed-device module can actually run. Today the
      VM rejects a dispatch whose device disagrees with the active device
- [ ] Blocked on op coverage: parallel-`linear` merge (needs `concat`/`slice`),
      transpose folding (needs `transpose`), reshape-as-metadata (needs
      `reshape`). `mul` is commented out in `HexirOps.td`, so a gated
      activation cannot be written at all

## 3. Kernel level — the `hextir` dialect

There is no `compiler/Dialect/HexTIR/Transforms/` yet. Creating it, with an
`add_subdirectory` in the parent, is the natural home for schedule passes.

- [ ] Reorder the matmul nest `i-j-k` → `i-k-j` for the CPU path. The FP result
      is bit-identical, since the `k` order per output element does not change
- [ ] Honour `unrolled` — clone the body for a constant extent (issue #10)
- [ ] Verify `kind` and reject `vectorized` until it works (issue #10)
- [ ] **`split` / `tile` as a pass** — rewrite one `hextir.for` into two nested
      ones. Every other schedule transform is built on this primitive
- [ ] **Fuse two `prim_func`s** — inline an elementwise consumer into the
      producer's loop body when the iteration spaces match, and drop the
      intermediate buffer. This is TVM's `FuseTIR`, and it is what makes the
      graph-level fusion decision real
- [ ] Hoist loop-invariant loads out of the reduction
- [ ] Better block size, and say something when it degenerates (issue #11)

### Dialect additions that unblock the rest

Small changes, each unblocking a whole class of schedules:

- [ ] `hextir.if` → bounds guards → tiles that do not divide the extent. Until
      this exists, `chooseBlockSize` is stuck on exact divisors and a dimension
      of 37 gets a one-thread block
- [ ] A memory space on `hextir.alloc_buffer`, plus a barrier op → shared-memory
      staging. Without both, `cache_read` is unexpressible and the matmul cannot
      get past roughly a third of cuBLAS
- [ ] Vector types in `buffer_load` / `buffer_store` → `vectorized` means
      something → 128-bit loads

## 4. Memory

See §1D: these are serializer and runtime changes, not passes.

- [ ] **Slot liveness and buffer reuse.** `ModuleSerializer::assignSlot` hands
      out `nextSlot++` and never reuses one, so every intermediate is a distinct
      allocation live for the whole program. Compute live ranges over the
      straight-line command list — it is a single basic block, so this is easy —
      and let non-overlapping values share a slot
- [ ] **One arena instead of per-dispatch allocation.** Each `ALLOC` becomes a
      `cuMemAlloc` in `vm_bind`, and the device allocator serializes against the
      driver. Allocate once at load, make slots offsets
- [ ] **Upload weights once.** `hexir_execute` re-runs every `CONST` on every
      call, so weights are re-uploaded per invocation. Needs a load phase
      separate from execute — a runtime API change, and the point at which
      repeated inference stops being silly
- [ ] **In-place elementwise.** `relu(x)` into `x`; `add` accumulating into an
      operand. Halves allocations and traffic for every activation. Needs an
      aliasing bit in `hexir_executable_entry_t` so the VM may legitimately
      pass one slot twice
- [ ] **A fill descriptor for splat constants.** The serializer expands a splat
      elementwise into RODATA, so a 256×256 constant of one value is 512 KB on
      disk and a full upload at runtime. It should be eight bytes and a command
- [ ] **Pinned staging.** `HEXIR_MEMORY_HOST_VISIBLE` exists in
      `runtime/include/hexir_runtime/hal.h` and the CUDA backend ignores it —
      everything is device-local. Pinned host memory makes H2D copies async and
      roughly twice as fast
- [ ] Free at last use, rather than everything at `vm_release`
- [ ] Drop the `malloc` per `PRINT` in `vm_print`
- [ ] Raise or remove the caps: `HEXIR_MAX_SLOTS` is 256 and a dispatch takes at
      most four arguments. Both are unreachable with today's op set and both are
      reached immediately by anything real. Until then, at least fail in the
      compiler with a clear message rather than emitting a module the runtime
      rejects with a bare `INVALID_MODULE`
- [ ] Linalg path only: in-place bufferization through `BufferizableOpInterface`.
      Note `IdentityLayoutMap` is forced during bufferization, so no view is
      ever non-contiguous and every layout change is a copy

## 5. Execution

- [ ] **Cache the loaded CUDA module and function** (issue #9). Pure overhead,
      no correctness risk, measurable with `bench/run.sh`
- [ ] **Stop synchronizing per dispatch.** `cuda_launch` calls
      `cuCtxSynchronize` before returning. Launch on a stream, sync only where a
      result is observed — which is `PRINT`, `END` and `hexir_buffer_read`, and
      nowhere else
- [ ] Command buffers in the HAL, then CUDA graph capture. The command list is
      static straight-line dataflow by construction, which is exactly the
      precondition graph capture wants
- [ ] Wire up the affine pass or reject the flag (issue #8)

## 6. Measuring

An optimization you cannot see is one you cannot keep.

```bash
hexir -emit=mlir -opt prog.mlir                 # graph rewrites
hexir -emit=mlir-tir prog.mlir                  # the kernel and its schedule
hexir -emit=mlir-linalg --print-ir-after-all    # which pass changed what
bench/run.sh                                    # end to end, against cuBLAS
tools/hxb-dump.py m.hxb                         # sections, commands, rodata
```

- [ ] `bench/hexir-bench.c` reports only the first executable. Add a
      per-dispatch breakdown, achieved bandwidth for memory-bound kernels, and a
      decode-shaped case (`M = 1`) — GFLOPS alone hides the entire elementwise
      story

Write tests with their own `%s` input rather than relying on the built-in
program: `Builder.cpp` uses `UnknownLoc`, so diagnostics have no location, and
its 2×2 shapes hide every scheduling bug.
