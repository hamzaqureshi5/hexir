# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Hexir (Heterogeneous EXecution IR) is a research MLIR compiler: a custom `hexir` dialect of
neural-network ops is progressively lowered to linalg → memref → SCF/CF → LLVM, with a
partitioning pass that assigns each op to CPU or CUDA. The CPU path JIT-executes in-process;
the CUDA path lowers to `gpu.launch` → NVVM → CUBIN (needs a CUDA toolkit).

## Build

`CMakeLists.txt` **hard-codes** the LLVM/MLIR install:

```cmake
set(LLVM_DIR "/home/user/llvm-project/build/lib/cmake/llvm")
set(MLIR_DIR "/home/user/llvm-project/build/lib/cmake/mlir")
```

Change those two lines (or pass `-DLLVM_DIR=/... -DMLIR_DIR=/...` and delete them) when the
MLIR build lives elsewhere. Everything is one `hexir` executable — there is no library target.

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)          # or ./build.sh from the repo root (cd build && make)
```

Sources are picked up by `file(GLOB ... CONFIGURE_DEPENDS)` over `src/*.cpp` and
`src/Dialects/*.cpp`, so new files need no CMake edit — but new TableGen files do
(`include/Dialects/CMakeLists.txt`). `targets/` is **not** part of the build: `add_subdirectory(targets)`
is absent and its CMakeLists references a nonexistent `ppytorch_core` — those backends are
inert scaffolding.

## Test

lit/FileCheck suite in `test/`. Requires `lit` (pip) and `FileCheck` (system LLVM; the config
searches PATH then `/usr/lib/llvm-{20,19,18,14}/bin`).

```bash
cd build && make check-hexir              # full suite
lit -v test                               # from repo root (auto-finds ./build)
lit -v test --param build_dir=/path/build
lit -v test/jit-cpu.mlir                  # single test
lit -v test --filter=jit
```

Tests that need the CUDA toolkit are gated `REQUIRES: cuda` (lit enables that feature when
`nvcc` is on PATH): `jit.mlir`, `llvm-ir.mlir`, `lower-to-llvm.mlir`.

**Two kinds of test, and it matters which you are writing.**

- `RUN: %hexir -emit=... %s` — compiles the file. `loadMLIR` parses `inputFilename` when it is
  not `-`. Diagnostics carry real source locations. Use this for anything dialect-level.
- `RUN: %hexir -emit=...` (no file) — compiles the program *built in C++* by
  `builder::createMLPLinearFunction` (`src/Builder.cpp`). All the emit-stage tests are this kind,
  which is why they assert on a 2x2 matmul nobody can see in the test file. To change what they
  compile, edit `Builder.cpp`. Everything is built with `UnknownLoc`, so errors have no location.

## Running

```bash
./build/hexir -emit=mlir            # hexir dialect (initial module)
./build/hexir -emit=mlir-linalg     # after LowerToLinalg
./build/hexir -emit=mlir-hetero     # after Partition + MaterializeLSTargets (ls_cpu/ls_gpu ops)
./build/hexir -emit=mlir-gpu        # cuda partitions as gpu.launch
./build/hexir -emit=mlir-llvm       # LLVM dialect
./build/hexir -emit=llvm            # translated LLVM IR
./build/hexir -emit=jit             # JIT and run @main
./build/hexir -emit=jit -opt        # + LLVM O3
./build/hexir -emit=mlir-hetero -placement=hexir.linear=cuda,hexir.relu=cpu
./build/hexir -emit=mlir-linalg --print-ir-after-all   # standard MLIR pass-manager flags work
```

All seven `-emit=` stages work; `-emit=jit` runs on the CPU with no CUDA toolkit installed and
prints `8 17 / 12 14`. `-emit=jit -placement=hexir.linear=cuda` needs the toolkit and fails on a
CPU-only box during `gpu-module-to-binary`.

`-emit=mlir-affine` exists in the `Action` enum but no affine pass is wired into the pipeline
(`createLowerToAffinePass` in `src/LowerToAffineLoops.cpp` is declared and built but never
added to a PassManager). `-emit=ast` is a stub.

## Architecture

The pipeline is assembled imperatively in `loadAndProcessMLIR` (`src/main.cpp`), which is the
single source of truth for pass ordering. Stage gating uses `emitAction >= Action::X` on the
**ordered** `Action` enum, so enum order is semantically load-bearing — reordering it changes
which passes run.

Pass order and the files implementing each:

1. `canonicalize` → `hexir-shape-inference` (`ShapeInferencePass.cpp`, via the
   `ShapeInferenceOpInterface`) → `canonicalize` → `cse`, nested on `hexir::FuncOp`
2. `hexir-partition` (`Partition.cpp`) — **runs before lowering**, annotating frontend `hexir.*`
   ops with `device = "cpu"|"cuda"` from the `TargetSupport` registry, and tags the module with
   `hexir.targets`
3. `hexir-lower-to-linalg` (`LowerToLinalg.cpp`) — `hexir.linear`→`linalg.matmul`,
   `hexir.relu`→`linalg.generic`+`arith.maximumf`, `hexir.constant`→`arith.constant`; each
   pattern **copies the `device` attr onto the op it creates**. `hexir.print` survives to the
   LLVM stage.
4. `hexir-partition` again — fallback for linalg ops with no inherited `device` (the pass skips
   any op that already has the attr, so propagated placement wins)
5. `hexir-materialize-ls-targets` (`MaterializeLSTargets.cpp`) — rewrites linalg ops into
   `ls_cpu.*`/`ls_gpu.*` model ops purely so placement is visible in the IR
6. `ls-lower-to-linalg` (`LowerLSToLinalg.cpp`) — converts them straight back to linalg,
   preserving `device`
7. `empty-tensor-to-alloc-tensor` → `one-shot-bufferize`
   (`unknownTypeConversion=IdentityLayoutMap`) → `canonicalize` → buffer-dealloc
   simplification (tensor → memref). The order matters: `tensor.empty` has no
   bufferization of its own and **must** be rewritten to
   `bufferization.alloc_tensor` first, or the alloc_tensor created mid-flight
   never enters the bufferization worklist and survives as an unbufferized op.
8. `hexir-lower-cuda-to-gpu` (`LowerCudaToGpu.cpp`) — for ops with `device == "cuda"`, builds a
   `gpu.launch` by hand (`lowerMatmul` for `linalg.matmul`, `lowerElementwise` otherwise)
9. `convert-linalg-to-loops` — must run **before** GPU lowering completes, since `gpu-to-llvm`
   cannot handle live linalg ops
10. CUDA-only: `gpu-kernel-outlining` → `nvvm-attach-target` (chip `sm_86`, `+ptx80`, hard-coded
    in `main.cpp`) → `convert-gpu-to-nvvm` (nested on `gpu::GPUModuleOp`) → `gpu-module-to-binary`
    (needs `ptxas`) → `gpu-to-llvm`
11. `convert-scf-to-cf` → `hexir-to-llvm` (`LowerToLLVM.cpp`, also lowers `hexir.print` to
    `printf` calls) → `reconcile-unrealized-casts` → LLVM DI scopes

Dialects (three of them):

- `hexir` — `include/Dialects/Ops.td` + `src/Dialects/Dialect.cpp`. Frontend NN ops. Many ops
  (`sigmoid`, `softmax`, `gelu`, `swish`, `mish`, `tanh`, `elu`, `leaky_relu`) are *declared in
  TableGen but have no lowering pattern* — only `constant`, `add`, `relu`, `linear`, `print`,
  and `hexir.func` are supported end to end.
- `hextir` — `include/Dialects/HexTIROps.td` + `src/Dialects/HexTIRDialect.cpp`. The **kernel
  level**: `prim_func` (destination-passing over memrefs, `FunctionOpInterface`), `block` (named
  schedulable region), `for` (with `kind` = serial/parallel/vectorized/unrolled/thread_binding —
  this attribute *is* the schedule), `alloc_buffer`, `buffer_load`/`buffer_store`, `yield`,
  `return`. Reached only via `hexir.call_tir`, whose verifier enforces the destination-passing
  contract: callee arity must be `args + 1`. Deliberately thin — it records scheduling decisions
  and hands loop nests to `scf`/`memref`/`gpu` rather than reimplementing a scheduling language.
  **Not yet wired into any pass pipeline**; it parses, verifies and round-trips, and that is all.
- `ls_cpu` / `ls_gpu` — `include/Dialects/LSDialects.td` + `src/Dialects/LSDialects.cpp`.
  Mirror-image `add`/`mul`/`matmul`/`relu` ops that exist only to make placement legible in
  `-emit=mlir-hetero`. Adding an op means adding it to *both* dialects plus a pattern in
  `MaterializeLSTargets.cpp` and `LowerLSToLinalg.cpp`.

Placement registry: `src/TargetInfo.cpp` — a singleton `TargetSupport` mapping op names to
supported targets (`opSupports_`) and a preferred target (`opPreferred_`). `"gpu"` normalizes to
`"cuda"`. Unregistered ops default to CPU. The `-placement <op>=<device>` flag calls
`setPreferredTarget` before any pass runs, and rejects op/device pairs absent from `opSupports_`.
Keys are frontend op names (`hexir.linear`), because partitioning happens pre-lowering.

Bufferization (`src/BufferizableOpInterfaceImpl.cpp`): One-Shot Bufferize only knows how to
convert ops that implement `BufferizableOpInterface`, and for most dialects that interface comes
from an **external model that has to be registered explicitly** in `main.cpp`. Hexir registers
five: `bufferization` (needed for its own `alloc_tensor`), `func_ext`, `arith`, `linalg`,
`tensor`, plus its own model for `hexir.print` (read-only, no results). A missing registration
shows up as the unhelpful `error: op was not bufferized` with no location, because
`src/Builder.cpp` builds everything with `UnknownLoc`. To find the offending op, run with
`--mlir-print-ir-after-failure` and look for what still has tensor operands — `to_tensor` and
`to_buffer` are allowed in the output, anything else is the culprit.

Do not hand-roll `bufferization.to_buffer` in an earlier pass to "pre-bufferize" an op: a module
that reaches One-Shot Bufferize already half-buffered is what caused that error historically.
Give the op an interface instead. Related: never use an `arith.constant` as a destination-passing
`outs` operand — a constant is read-only, so bufferization has to emit a read-only global plus an
`alloc_tensor(copy)` to make it writable. Use `tensor.empty` (+ `linalg.fill` when the op
accumulates rather than fully overwrites).

TableGen: `include/Dialects/CMakeLists.txt` generates ops/dialect/interface `.inc` files into
`build/include/Dialects/`; `src/HexirCombine.td` generates `build/HexirCombine.inc` (DRR
rewrites — currently all patterns are commented out; only `ConstantOp::fold` is live).

## Gotchas

- **`src/Jit.cpp` is dead code.** It defines a global `runJit`, but `main.cpp` defines and calls
  its own `static runJit`. Edits to `Jit.cpp` have no effect on the binary.
- **5 of 10 checked-in tests fail, for two reasons unrelated to the compiler.**
  `TargetInfo.cpp` sets `opPreferred_["hexir.linear"] = "cpu"` (with a comment claiming GPU), and
  the relu call in `createMLPLinearFunction` is commented out — so `-emit=mlir-hetero` emits
  `ls_cpu.matmul` and no relu, while `partition-hetero.mlir`, `placement-flag.mlir`
  (DEFAULT/SWAP/ALLGPU), `hexir-dialect.mlir`, `lower-to-linalg.mlir`, and `cuda-to-gpu.mlir`
  expect `ls_gpu.matmul` and a relu. Restoring the relu fixes the first two; flipping the default
  to `cuda` fixes the rest but makes plain `-emit=jit` require a CUDA toolkit.
  `jit-cpu.mlir` and `hextir-roundtrip.mlir` pass; the other 3 are `REQUIRES: cuda` and report as
  unsupported.
- `FileCheck` is not on PATH on this machine; it lives in the LLVM build tree
  (`/home/user/llvm-project/build/bin`), which `test/lit.cfg.py` does not search. Prepend it to
  PATH to run the suite. `lit` is importable as a Python module but has no console script, so
  `python3 -m lit` fails — call `lit.main.main()` directly or use the `check-hexir` target.
- The JIT loads `/usr/local/lib/libmlir_c_runner_utils.so` and
  `libmlir_cuda_runtime.so` only if they exist (`main.cpp::runJit`); neither is present on this
  machine and the CPU path works anyway, since `printf` resolves from libc.
- `src/` still carries Toy-tutorial provenance in comments and file headers; naming is
  inconsistent (`hexir::FuncOp` vs `func::FuncOp` — the shape-inference pass nests on the
  former, which only exists if `hexir.func` ops are built).
- `docs/cuda-server.md` is the runbook for the A6000/sm_86 setup and the failure/symptom table.
