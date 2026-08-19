# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Hexir (Heterogeneous EXecution IR) is a research MLIR compiler: a custom `hexir` dialect of
neural-network ops is progressively lowered to linalg → memref → SCF/CF → LLVM, with a
partitioning pass that assigns each op to CPU or CUDA. The CPU path JIT-executes in-process;
the CUDA path lowers to `gpu.launch` → NVVM → CUBIN (needs a CUDA toolkit).

## Build

`CMakeLists.txt` defaults the LLVM/MLIR install to `/home/user/llvm-project/build/...` but only
`if(NOT DEFINED ...)`, so `-DLLVM_DIR=/... -DMLIR_DIR=/...` overrides it without editing the file.

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)          # or ./build.sh from the repo root (cd build && make)
```

The binary is pinned to `build/hexir` (`RUNTIME_OUTPUT_DIRECTORY` in
`tools/hexir/CMakeLists.txt`) because `test/lit.cfg.py`, the readme and docs all refer to it.

**Layout** (MLIR/IREE convention: public headers and TableGen under `include/hexir/`,
implementations mirrored under `lib/`):

```
include/hexir/
  Dialect/{Hexir,HexTIR,LS}/IR/         dialect headers + .td  (one CMakeLists each, tablegen)
  Dialect/{Hexir,LS}/Transforms/        Passes.h for same-dialect rewrites
  Conversion/Passes.h                   every dialect-to-dialect pass
  Pipelines/Pipelines.h                 the pass pipeline, and the Stage enum
  Target/, Support/
lib/
  Dialect/<Name>/{IR,Transforms}/       mirrors include/
  Conversion/<A>To<B>/                  one directory per conversion
  Pipelines/                            pass ordering
  Target/, Support/
tools/hexir/hexir.cpp                   CLI, LLVM translation, JIT
runtime/                                standalone C runtime (no MLIR/LLVM)
cmake/HexirLibrary.cmake                hexir_library() helper
```

**There is no glob.** Each directory has a `CMakeLists.txt` calling `hexir_library(<name> ...)`,
which builds a static lib, wires its TableGen `DEPENDS`, and appends to the `HEXIR_LIBS` global
property. `tools/hexir` links whatever is in that property inside `--start-group`. So a new file
needs a line in its directory's `CMakeLists.txt`, and a new directory needs an `add_subdirectory`
in its parent. New TableGen files need an entry in the matching `include/.../IR/CMakeLists.txt`.

Every hexir library links the same `HEXIR_MLIR_LIBS` set from the root `CMakeLists.txt`. The
layering that is enforced here is between hexir's own modules; pinning exact MLIR deps per
library would be churn at this size.

### Runtime

`runtime/` is a separate C99 project — its own `CMakeLists.txt`, buildable standalone, and with
**no include path into `include/` and no MLIR or LLVM dependency**. That is the point: deploying a
compiled module must not mean shipping the compiler. It is added from the root `CMakeLists.txt`
*before* the LLVM/MLIR `include_directories()` so it cannot inherit them. If it ever needs to link
an MLIR library, the layering is wrong.

```
runtime/include/hexir_runtime/   status.h  hal.h  module.h  runtime.h
runtime/src/hal/                 vtable dispatch + cpu/ backend
runtime/src/module/              mmap loader for the .hxb container
runtime/tools/hexir-run/         deploy-side CLI -> build/hexir-run
```

The container is a header, a section table, then 8-byte-aligned payloads
(SYMBOLS / PROGRAM / RODATA / EXECUTABLES). Sections are found by absolute file offset and used
in place from the mapping, so a large RODATA costs nothing to load. Offsets come off disk and are
bounds-checked before any pointer is handed out.

**What is stubbed:** the compiler has no `-emit=hxb`, so nothing produces a module yet, and
`hexir-run` has no command-list interpreter — it loads, validates, reports sections, and exits
`HEXIR_ERROR_UNIMPLEMENTED` if asked to run. `--selftest` exercises the HAL without a module. The
CUDA backend returns `HEXIR_ERROR_UNIMPLEMENTED`; it belongs in `src/hal/cuda/` and should use the
driver API (`libcuda`) rather than the CUDA runtime API so it can be dlopened.

`targets/` is **not** part of the build and references a nonexistent `ppytorch_core` — inert
scaffolding. Per the architecture discussion those directories are the natural home for runtime
HAL backends, not compiler backends.

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
  `builder::createMLPLinearFunction` (`lib/Support/Builder.cpp`). All the emit-stage tests are this kind,
  which is why they assert on a 2x2 matmul nobody can see in the test file. To change what they
  compile, edit `Builder.cpp`. Everything is built with `UnknownLoc`, so errors have no location.

## Running

```bash
./build/hexir -emit=mlir            # hexir dialect (initial module)
./build/hexir -emit=mlir-tir        # after LowerToTIR (hextir prim funcs + call_tir)
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
(`createLowerToAffinePass` in `lib/Conversion/HexirToAffine/` is declared and built but never
added to a PassManager). `-emit=ast` is a stub.

## Architecture

Pass ordering lives in `buildHexirPipeline` (`lib/Pipelines/Pipelines.cpp`) — the single source of
truth. It only *adds* passes and returns early once it has reached the requested stage; the driver
runs the `PassManager` exactly once. Stage gating uses `>=` on the **ordered** `hexir::Stage` enum
(`include/hexir/Pipelines/Pipelines.h`), so enum order is semantically load-bearing — reordering
it changes which passes run. `tools/hexir/hexir.cpp` is now only CLI parsing, LLVM translation and
the JIT.

Pass order and the files implementing each:

1. `canonicalize` → `hexir-shape-inference` (`Dialect/Hexir/Transforms/ShapeInference.cpp`, via the
   `ShapeInferenceOpInterface`) → `canonicalize` → `cse`, nested on `hexir::FuncOp`
1b. `-emit=mlir-tir` branches off here: `hexir-partition` → `hexir-lower-to-tir`
   (`Conversion/HexirToTIR/`), then stops. Each `hexir.linear`/`add`/`relu` becomes a
   `hextir.prim_func` at module scope plus a `hexir.call_tir`. The `device` attr picks the loop
   kinds — `cpu` → `parallel`, `cuda` → `thread_binding` bound to `blockIdx.x`/`threadIdx.x`;
   reduction axes stay `serial`. `DumpMLIRTIR` sits below `DumpMLIRLinalg` in the `Action` enum
   precisely so `emitAction >= DumpMLIRLinalg` stays false and the linalg pipeline does not run.
2. `hexir-partition` (`Dialect/Hexir/Transforms/Partition.cpp`) — **runs before lowering**, annotating frontend `hexir.*`
   ops with `device = "cpu"|"cuda"` from the `TargetSupport` registry, and tags the module with
   `hexir.targets`
3. `hexir-lower-to-linalg` (`Conversion/HexirToLinalg/`) — `hexir.linear`→`linalg.matmul`,
   `hexir.relu`→`linalg.generic`+`arith.maximumf`, `hexir.constant`→`arith.constant`; each
   pattern **copies the `device` attr onto the op it creates**. `hexir.print` survives to the
   LLVM stage.
4. `hexir-partition` again — fallback for linalg ops with no inherited `device` (the pass skips
   any op that already has the attr, so propagated placement wins)
5. `hexir-materialize-ls-targets` (`Dialect/LS/Transforms/MaterializeLSTargets.cpp`) — rewrites linalg ops into
   `ls_cpu.*`/`ls_gpu.*` model ops purely so placement is visible in the IR
6. `ls-lower-to-linalg` (`Conversion/LSToLinalg/`) — converts them straight back to linalg,
   preserving `device`
7. `empty-tensor-to-alloc-tensor` → `one-shot-bufferize`
   (`unknownTypeConversion=IdentityLayoutMap`) → `canonicalize` → buffer-dealloc
   simplification (tensor → memref). The order matters: `tensor.empty` has no
   bufferization of its own and **must** be rewritten to
   `bufferization.alloc_tensor` first, or the alloc_tensor created mid-flight
   never enters the bufferization worklist and survives as an unbufferized op.
8. `hexir-lower-cuda-to-gpu` (`Conversion/CudaToGpu/`) — for ops with `device == "cuda"`, builds a
   `gpu.launch` by hand (`lowerMatmul` for `linalg.matmul`, `lowerElementwise` otherwise)
9. `convert-linalg-to-loops` — must run **before** GPU lowering completes, since `gpu-to-llvm`
   cannot handle live linalg ops
10. CUDA-only: `gpu-kernel-outlining` → `nvvm-attach-target` (chip `sm_86`, `+ptx80`, hard-coded
    in `main.cpp`) → `convert-gpu-to-nvvm` (nested on `gpu::GPUModuleOp`) → `gpu-module-to-binary`
    (needs `ptxas`) → `gpu-to-llvm`
11. `convert-scf-to-cf` → `hexir-to-llvm` (`Conversion/HexirToLLVM/`, also lowers `hexir.print` to
    `printf` calls) → `reconcile-unrealized-casts` → LLVM DI scopes

Dialects (three of them):

- `hexir` — `include/hexir/Dialect/Hexir/IR/HexirOps.td` + `lib/Dialect/Hexir/IR/HexirDialect.cpp`. Frontend NN ops. Many ops
  (`sigmoid`, `softmax`, `gelu`, `swish`, `mish`, `tanh`, `elu`, `leaky_relu`) are *declared in
  TableGen but have no lowering pattern* — only `constant`, `add`, `relu`, `linear`, `print`,
  and `hexir.func` are supported end to end.
- `hextir` — `include/hexir/Dialect/HexTIR/IR/HexTIROps.td` + `lib/Dialect/HexTIR/IR/HexTIRDialect.cpp`. The **kernel
  level**: `prim_func` (destination-passing over memrefs, `FunctionOpInterface`), `block` (named
  schedulable region), `for` (with `kind` = serial/parallel/vectorized/unrolled/thread_binding —
  this attribute *is* the schedule), `alloc_buffer`, `buffer_load`/`buffer_store`, `yield`,
  `return`. Reached only via `hexir.call_tir`, whose verifier enforces the destination-passing
  contract: callee arity must be `args + 1`. Deliberately thin — it records scheduling decisions
  and hands loop nests to `scf`/`memref`/`gpu` rather than reimplementing a scheduling language.
  Produced by `hexir-lower-to-tir` (`-emit=mlir-tir`). Nothing lowers *out* of it yet, so that
  stage is terminal.
- `ls_cpu` / `ls_gpu` — `include/hexir/Dialect/LS/IR/LSDialects.td` + `lib/Dialect/LS/IR/LSDialects.cpp`.
  Mirror-image `add`/`mul`/`matmul`/`relu` ops that exist only to make placement legible in
  `-emit=mlir-hetero`. Adding an op means adding it to *both* dialects plus a pattern in
  `MaterializeLSTargets.cpp` and `LowerLSToLinalg.cpp`.

Placement registry: `lib/Target/TargetInfo.cpp` — a singleton `TargetSupport` mapping op names to
supported targets (`opSupports_`) and a preferred target (`opPreferred_`). `"gpu"` normalizes to
`"cuda"`. Unregistered ops default to CPU. The `-placement <op>=<device>` flag calls
`setPreferredTarget` before any pass runs, and rejects op/device pairs absent from `opSupports_`.
Keys are frontend op names (`hexir.linear`), because partitioning happens pre-lowering.

Bufferization (`lib/Support/BufferizableOpInterfaceImpl.cpp`): One-Shot Bufferize only knows how to
convert ops that implement `BufferizableOpInterface`, and for most dialects that interface comes
from an **external model that has to be registered explicitly** in `main.cpp`. Hexir registers
five: `bufferization` (needed for its own `alloc_tensor`), `func_ext`, `arith`, `linalg`,
`tensor`, plus its own model for `hexir.print` (read-only, no results). A missing registration
shows up as the unhelpful `error: op was not bufferized` with no location, because
`lib/Support/Builder.cpp` builds everything with `UnknownLoc`. To find the offending op, run with
`--mlir-print-ir-after-failure` and look for what still has tensor operands — `to_tensor` and
`to_buffer` are allowed in the output, anything else is the culprit.

Do not hand-roll `bufferization.to_buffer` in an earlier pass to "pre-bufferize" an op: a module
that reaches One-Shot Bufferize already half-buffered is what caused that error historically.
Give the op an interface instead. Related: never use an `arith.constant` as a destination-passing
`outs` operand — a constant is read-only, so bufferization has to emit a read-only global plus an
`alloc_tensor(copy)` to make it writable. Use `tensor.empty` (+ `linalg.fill` when the op
accumulates rather than fully overwrites).

TableGen: each `include/hexir/Dialect/*/IR/CMakeLists.txt` generates its `.inc` files into the mirrored
`build/include/hexir/...` path, included by that full path; `HexirCombine.td` generates `HexirCombine.inc` (DRR
rewrites — currently all patterns are commented out; only `ConstantOp::fold` is live).

## Gotchas

- **5 of 11 checked-in tests fail, for two reasons unrelated to the compiler.**
  `TargetInfo.cpp` sets `opPreferred_["hexir.linear"] = "cpu"` (with a comment claiming GPU), and
  the relu call in `createMLPLinearFunction` is commented out — so `-emit=mlir-hetero` emits
  `ls_cpu.matmul` and no relu, while `partition-hetero.mlir`, `placement-flag.mlir`
  (DEFAULT/SWAP/ALLGPU), `hexir-dialect.mlir`, `lower-to-linalg.mlir`, and `cuda-to-gpu.mlir`
  expect `ls_gpu.matmul` and a relu. Restoring the relu fixes the first two; flipping the default
  to `cuda` fixes the rest but makes plain `-emit=jit` require a CUDA toolkit.
  `jit-cpu.mlir`, `hextir-roundtrip.mlir` and `lower-to-tir.mlir` pass; the other 3 are
  `REQUIRES: cuda` and report as unsupported.
- **Do not write a FileCheck prefix followed by a colon in test prose.** A comment like
  `// Everything on the CPU: ...` in a test using `--check-prefix=CPU` is parsed as a directive
  and fails with a confusing "expected string not found in input".
- `FileCheck` is not on PATH on this machine; it lives in the LLVM build tree
  (`/home/user/llvm-project/build/bin`), which `test/lit.cfg.py` does not search. Prepend it to
  PATH to run the suite. `lit` is importable as a Python module but has no console script, so
  `python3 -m lit` fails — call `lit.main.main()` directly or use the `check-hexir` target.
- The JIT loads `/usr/local/lib/libmlir_c_runner_utils.so` and
  `libmlir_cuda_runtime.so` only if they exist (`main.cpp::runJit`); neither is present on this
  machine and the CPU path works anyway, since `printf` resolves from libc.
- `lib/` still carries Toy-tutorial provenance in comments and file headers; naming is
  inconsistent (`hexir::FuncOp` vs `func::FuncOp` — the shape-inference pass nests on the
  former, which only exists if `hexir.func` ops are built).
- `docs/cuda-server.md` is the runbook for the A6000/sm_86 setup and the failure/symptom table.
