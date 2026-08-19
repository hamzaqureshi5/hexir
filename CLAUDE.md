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
implementations mirrored under `compiler/`):

```
include/hexir/
  Dialect/{Hexir,HexTIR,LS}/IR/         dialect headers + .td  (one CMakeLists each, tablegen)
  Dialect/{Hexir,LS}/Transforms/        Passes.h for same-dialect rewrites
  Conversion/Passes.h                   every dialect-to-dialect pass
  Pipelines/Pipelines.h                 the pass pipeline, and the Stage enum
  Serialization/ModuleSerializer.h      emit a loadable module
  Target/, Support/
compiler/
  Dialect/<Name>/{IR,Transforms}/       mirrors include/
  Conversion/<A>To<B>/                  one directory per conversion
  Pipelines/                            pass ordering
  Serialization/                        .hxb writer (includes runtime headers)
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

`runtime/include/hexir_runtime/program.h` defines the PROGRAM/EXECUTABLES encoding and is the
**contract between the two halves**. It lives on the runtime side deliberately: the compiler emits
the format the runtime defines, and `compiler/Serialization` includes it. Headers only — nothing links
back, and the dependency must never point the other way.

The host program is a flat command list (`ALLOC`/`CONST`/`DISPATCH`/`PRINT`/`END`), not bytecode:
everything the compiler can currently produce is straight-line dataflow, so an interpreter with
branches would be unused machinery. Bytecode becomes necessary with dynamic shapes or control
flow. Buffers are referred to by slot, a dense index space `ModuleSerializer` assigns.

**cuda kernels now carry real device code; cpu kernels are still descriptors.** For a
cuda-placed kernel, `-emit=hxb` lowers `hextir` → `gpu` → NVVM → CUBIN (`compiler/Serialization`
runs that pipeline on a *clone*, since the module being serialized must stay at the kernel level)
and embeds the image, so `hexir-run --device=cuda` executes it on the GPU with no compiler
present. Kernels use **bare-pointer calling convention**, so a kernel argument is one device
address rather than a seven-scalar memref descriptor. A cpu kernel still carries only a
descriptor The runtime supplies the bodies from
`src/kernels/reference_kernels.c`. That was the way to make the container, loader, HAL and command
list real and testable end to end; embedding a CUBIN from `gpu.binary`, or a host object, replaces
the descriptors without changing the container or the command list. Until then a module is
portable but not actually carrying compiled code. The CUDA HAL backend also returns
`HEXIR_ERROR_UNIMPLEMENTED`; it belongs in `src/hal/cuda/` using the driver API (`libcuda`) rather
than the CUDA runtime API, so it can be dlopened.

`hexir-run --selftest` exercises the HAL without needing a module.

`tools/hxb-dump.py` reads a module and prints its sections, disassembles the command list, shows
rodata as f64, and with `--image <kernel>` extracts the device image (then `cuobjdump --dump-sass`
on it). It is a deliberate second implementation of the format reader — when it and the runtime
disagree, one of them is wrong.

`targets/` is **not** part of the build and references a nonexistent `ppytorch_core` — inert
scaffolding. Per the architecture discussion those directories are the natural home for runtime
HAL backends, not compiler backends.

## Test

lit/FileCheck suite in `test/`. Requires `lit` (pip) and `FileCheck` (system LLVM; the config
searches PATH then `/usr/lib/llvm-{20,19,18,14}/bin`).

The compiler and the runtime each own their tests:

```bash
cd build && make check-hexir              # both suites, one lit run
cd build && make check-hexir-compiler     # compiler/test only
cd build && make check-hexir-runtime      # runtime/test only

lit -v compiler/test runtime/test         # from repo root (auto-finds ./build)
lit -v runtime/test --param build_dir=/path/build
lit -v compiler/test/jit-cpu.mlir         # single test
```

`compiler/test/` needs only `hexir`. `runtime/test/` runs `hexir-run`, and holds the artifact
tests — they invoke `hexir` too, but the behaviour under test is the runtime's, and a test belongs
with the binary whose behaviour it asserts.

Both read `lit.common.cfg.py` at the repo root via `lit_config.load_config`, which does all the
build-dir and tool discovery so the two cannot drift. That function runs the file in its own
namespace, so anything a suite needs is hung off `config` (`config.hexir_binary`,
`config.hexir_run_binary`, `config.hexir_build_dir`).

Tests that need the CUDA toolkit are gated `REQUIRES: cuda` (lit enables that feature when
`nvcc` is on PATH): `jit.mlir`, `llvm-ir.mlir`, `lower-to-llvm.mlir`.

**Two kinds of test, and it matters which you are writing.**

- `RUN: %hexir -emit=... %s` — compiles the file. `loadMLIR` parses `inputFilename` when it is
  not `-`. Diagnostics carry real source locations. Use this for anything dialect-level.
- `RUN: %hexir -emit=...` (no file) — compiles the program *built in C++* by
  `builder::createMLPLinearFunction` (`compiler/Support/Builder.cpp`). All the emit-stage tests are this kind,
  which is why they assert on a 2x2 matmul nobody can see in the test file. To change what they
  compile, edit `Builder.cpp`. Everything is built with `UnknownLoc`, so errors have no location.

## Running

```bash
./build/hexir -emit=mlir            # hexir dialect (initial module)
./build/hexir -emit=mlir-tir        # after LowerToTIR (hextir prim funcs + call_tir)
./build/hexir -emit=hxb -o m.hxb    # loadable module for the runtime
./build/hexir-run m.hxb             # run it, with no MLIR/LLVM in the process
./build/hexir -emit=mlir-linalg     # after LowerToLinalg
./build/hexir -emit=mlir-hetero     # after Partition (linalg ops carrying `device`)
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
(`createLowerToAffinePass` in `compiler/Conversion/HexirToAffine/` is declared and built but never
added to a PassManager). `-emit=ast` is a stub.

## Architecture

Pass ordering lives in `buildHexirPipeline` (`compiler/Pipelines/Pipelines.cpp`) — the single source of
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
5. `-emit=mlir-hetero` stops here: linalg carrying `device` attributes. A detour through
   mirror `ls_cpu`/`ls_gpu` ops used to exist purely to show placement in the op name; it was
   removed because the round trip rebuilt each op, dropping its `outs` operand and every
   attribute but `device`
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

Dialects (two of them):

- `hexir` — `include/hexir/Dialect/Hexir/IR/HexirOps.td` + `compiler/Dialect/Hexir/IR/HexirDialect.cpp`. Frontend NN ops. Many ops
  (`sigmoid`, `softmax`, `gelu`, `swish`, `mish`, `tanh`, `elu`, `leaky_relu`) are *declared in
  TableGen but have no lowering pattern* — only `constant`, `add`, `relu`, `linear`, `print`,
  and `hexir.func` are supported end to end.
- `hextir` — `include/hexir/Dialect/HexTIR/IR/HexTIROps.td` + `compiler/Dialect/HexTIR/IR/HexTIRDialect.cpp`. The **kernel
  level**: `prim_func` (destination-passing over memrefs, `FunctionOpInterface`), `block` (named
  schedulable region), `for` (with `kind` = serial/parallel/vectorized/unrolled/thread_binding —
  this attribute *is* the schedule), `alloc_buffer`, `buffer_load`/`buffer_store`, `yield`,
  `return`. Reached only via `hexir.call_tir`, whose verifier enforces the destination-passing
  contract: callee arity must be `args + 1`. Deliberately thin — it records scheduling decisions
  and hands loop nests to `scf`/`memref`/`gpu` rather than reimplementing a scheduling language.
  Produced by `hexir-lower-to-tir` (`-emit=mlir-tir`). Nothing lowers *out* of it yet, so that
  stage is terminal.

Placement registry: `compiler/Target/TargetInfo.cpp` — a singleton `TargetSupport` mapping op names to
supported targets (`opSupports_`) and a preferred target (`opPreferred_`). `"gpu"` normalizes to
`"cuda"`. Unregistered ops default to CPU. The `-placement <op>=<device>` flag calls
`setPreferredTarget` before any pass runs, and rejects op/device pairs absent from `opSupports_`.
Keys are frontend op names (`hexir.linear`), because partitioning happens pre-lowering.

Bufferization (`compiler/Support/BufferizableOpInterfaceImpl.cpp`): One-Shot Bufferize only knows how to
convert ops that implement `BufferizableOpInterface`, and for most dialects that interface comes
from an **external model that has to be registered explicitly** in `main.cpp`. Hexir registers
five: `bufferization` (needed for its own `alloc_tensor`), `func_ext`, `arith`, `linalg`,
`tensor`, plus its own model for `hexir.print` (read-only, no results). A missing registration
shows up as the unhelpful `error: op was not bufferized` with no location, because
`compiler/Support/Builder.cpp` builds everything with `UnknownLoc`. To find the offending op, run with
`--mlir-print-ir-after-failure` and look for what still has tensor operands — `to_tensor` and
`to_buffer` are allowed in the output, anything else is the culprit.

`PrintOpInterface::bufferize` copies the op's attributes onto the op it creates.
`replaceOpWithNewBufferizedOp` builds a fresh op, so anything not passed as an operand is dropped
— including `device`, which every later pass needs. Any rewrite that rebuilds an op must carry
attributes across; that failure mode is what made the old mirror dialects lossy.

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

- **5 of 14 checked-in tests fail (all in the compiler suite; the runtime suite is green), for two reasons unrelated to the compiler.**
  `TargetInfo.cpp` sets `opPreferred_["hexir.linear"] = "cpu"` (with a comment claiming GPU), and
  the relu call in `createMLPLinearFunction` is commented out. Both remaining causes bite only the
  tests that compile the *built-in* program rather than their own input file:
  `hexir-dialect.mlir` and `lower-to-linalg.mlir` expect a relu that is not built, and
  `cuda-to-gpu.mlir` expects `gpu.launch` by default. The fix used for `partition-hetero.mlir` and
  `placement-flag.mlir` applies: give the test its own `%s` input and pass `-placement` explicitly.
  Passing: `jit-cpu`, `hextir-roundtrip`, `lower-to-tir`, `partition-hetero`, `placement-flag`,
  plus all 3 runtime tests. The other 3 are `REQUIRES: cuda` and report as unsupported.
- **Do not write a FileCheck prefix followed by a colon in test prose.** A comment like
  `// Everything on the CPU: ...` in a test using `--check-prefix=CPU` is parsed as a directive
  and fails with a confusing "expected string not found in input".
- **Never write a lit directive name followed by a colon in test prose.** `UNSUPPORTED:`,
  `REQUIRES:` and `XFAIL:` are parsed wherever they appear, including inside an explanatory
  comment, and the test fails as UNRESOLVED with a confusing parse error. The same applies to a
  FileCheck prefix: `// Everything on the CPU: ...` in a test using `--check-prefix=CPU` is read as
  a directive.
- Tests asserting a non-zero exit need `not` in front of the command, or lit fails the RUN line
  itself. `%hexir-run` is registered in `lit.cfg.py` *before* `%hexir`, since lit applies
  substitutions in list order and the shorter pattern would otherwise eat the longer one.
- `FileCheck` is not on PATH on this machine; it lives in the LLVM build tree
  (`/home/user/llvm-project/build/bin`), which `test/lit.cfg.py` does not search. Prepend it to
  PATH to run the suite. `lit` is importable as a Python module but has no console script, so
  `python3 -m lit` fails — call `lit.main.main()` directly or use the `check-hexir` target.
- The JIT loads `/usr/local/lib/libmlir_c_runner_utils.so` and
  `libmlir_cuda_runtime.so` only if they exist (`main.cpp::runJit`); neither is present on this
  machine and the CPU path works anyway, since `printf` resolves from libc.
- `compiler/` still carries Toy-tutorial provenance in comments and file headers; naming is
  inconsistent (`hexir::FuncOp` vs `func::FuncOp` — the shape-inference pass nests on the
  former, which only exists if `hexir.func` ops are built).
- **Docs are a Sphinx site in `docs/`**, published to GitHub Pages by
  `.github/workflows/docs.yml`. Pages are MyST Markdown so they read the same on GitHub and on the
  site; diagrams are ```` ```mermaid ```` fences turned into directives by `myst_fence_as_directive`.
  Build locally with `pip install -r docs/requirements.txt && sphinx-build -W -b html docs docs/_build/html`.
  CI uses `-W`, so a broken link or a page missing from a toctree fails the build. Pygments has no
  MLIR lexer, so `misc.highlighting_failure` is suppressed in `conf.py`.
- **`libdevice` is found automatically** (`compiler/Target/CudaToolkit.cpp`). MLIR looks for it at
  exactly `<toolkit>/nvvm/libdevice/libdevice.10.bc`; Ubuntu's package puts it in
  `/usr/lib/nvidia-cuda-toolkit/libdevice/` with no `nvvm` directory, giving
  `error: LibDevice path: /usr/nvvm/... does not exist` — which reads like a broken CUDA install
  and is not one. Hexir locates libdevice and builds a symlink directory under the user cache that
  satisfies the lookup. `CUDA_ROOT`/`CUDA_HOME`/`CUDA_PATH` override it.
- `docs/cuda-server.md` is the runbook for the A6000/sm_86 setup and the failure/symptom table.
