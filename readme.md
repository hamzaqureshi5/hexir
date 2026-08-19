<div align="center">
  <img src="assets/logo.svg" width="140" alt="Hexir logo"/>

  # Hexir

  **A heterogeneous ML compiler built on MLIR**

  *One graph in — partitioned, lowered, and executed across CPU and GPU.*

  [![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
  [![MLIR](https://img.shields.io/badge/LLVM-MLIR-orange.svg)](https://mlir.llvm.org/)
  [![CMake](https://img.shields.io/badge/CMake-3.13.4%2B-064F8C.svg)](https://cmake.org/)
  [![Tests](https://img.shields.io/badge/tests-15%2F15-brightgreen.svg)](#testing)
  [![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

  [Documentation](https://hamzaqureshi5.github.io/hexir/) ·
  [Getting started](https://hamzaqureshi5.github.io/hexir/getting-started.html) ·
  [Architecture](https://hamzaqureshi5.github.io/hexir/overview.html)

</div>

---

Hexir (**H**eterogeneous **EX**ecution **IR**) compiles neural-network programs
through a custom MLIR dialect, decides per operation whether it runs on the CPU
or the GPU, and either executes it in-process or writes a deployable file.

It is small enough to read. That is deliberate: the gap between MLIR's Toy
tutorial and a production compiler like IREE is enormous, and Hexir sits in the
middle — a complete pipeline, in about five thousand lines.

```mlir
%m = hexir.linear %a, %w : tensor<2x2xf64>   // placed on the GPU
%r = hexir.relu %m : tensor<2x2xf64>         // placed on the CPU
hexir.print %r : tensor<2x2xf64>
```

## Two ways to run a program

```bash
hexir -emit=jit                              # compile and run, in one process
```

```bash
hexir -emit=hxb -o model.hxb                 # compile to a file
hexir-run model.hxb                          # run it later, no compiler present
```

`hexir-run` links no MLIR and no LLVM. A GPU-placed kernel is compiled to a
CUBIN and embedded in the file, so the runtime loads and launches it with no
compiler in the process:

```console
$ hexir -emit=hxb -o gpu.hxb -placement=hexir.linear=cuda
$ hexir-run --device=cuda gpu.hxb
device        : cuda (NVIDIA GeForce GTX 1660 Ti)
--
8.000000 17.000000
12.000000 14.000000
```

Identical to the CPU answer, which is the point.

## How it works

Five levels, each answering a different question about the same program.

```
      hexir dialect        what to compute
            │
      partitioning         where each operation runs   (device = cpu | cuda)
            │
      hextir dialect       how one device computes it  (loops, buffers)
            │
   linalg / memref / scf   the actual loop nests
            │
      LLVM  /  NVVM        machine code
```

Placement is decided **before** lowering and travels down with the operation,
so by the time GPU code generation runs, the ops that should become kernels are
already labelled. At the kernel level it stops being an annotation and becomes
code: a CPU op gets `parallel` loops, a GPU op gets loops bound to `blockIdx`
and `threadIdx`.

Every stage can be printed, which is the most useful thing about the project
for learning:

| Command | Shows |
| --- | --- |
| `-emit=mlir` | the graph you wrote |
| `-emit=mlir-tir` | each operation as a kernel |
| `-emit=mlir-linalg` | loops on tensors |
| `-emit=mlir-hetero` | placement, as `device` attributes |
| `-emit=mlir-gpu` | CUDA kernels as `gpu.launch` |
| `-emit=llvm` | LLVM IR |
| `-emit=hxb` | a deployable module |
| `-emit=jit` | compile and run |

Move an operation between devices without recompiling the compiler:

```bash
hexir -emit=mlir-tir -placement=hexir.linear=cuda,hexir.relu=cpu
```

## Building

Needs a build of LLVM with MLIR.

```bash
git clone git@github.com:hamzaqureshi5/hexir.git && cd hexir
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
         -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
make -j$(nproc)
```

For the GPU path, LLVM must be built with `-DLLVM_TARGETS_TO_BUILD="X86;NVPTX"`
and `-DMLIR_ENABLE_CUDA_RUNNER=ON`, and a CUDA toolkit must be installed. Full
commands are in [`docs/llvm-cuda-build.txt`](docs/llvm-cuda-build.txt).

## Testing

```bash
cd build && make check-hexir              # everything
cd build && make check-hexir-compiler     # compiler only
cd build && make check-hexir-runtime      # runtime only
```

The compiler and the runtime each own their tests, in `compiler/test/` and
`runtime/test/`.

The runtime suite also runs standalone, against a module checked in at
`runtime/test/fixtures/`, so it needs no MLIR:

```bash
cmake -S runtime -B build-runtime && cmake --build build-runtime
lit runtime/test --param build_dir=$PWD/build-runtime
```

That is what CI runs. The compiler is not covered yet: it needs an unreleased
LLVM, so no packaged MLIR can build it.

## Layout

```
compiler/    Dialect/<Name>/{IR,Transforms}   dialects and their own passes
             Conversion/<A>To<B>/             one directory per lowering
             Pipelines/ Serialization/        pass order, the .hxb writer
runtime/     include/ src/hal/ src/vm/        standalone C, no MLIR or LLVM
tools/       hexir/  hxb-dump.py              the driver, and a module inspector
docs/        Sphinx sources
```

## Status

Research software. Some of it is finished, some is scaffolding, and the docs
say which rather than leaving you to find out.

| Works | Partly | Not yet |
| --- | --- | --- |
| CPU path, end to end | GPU kernels are one block, one thread | Transfer insertion in the JIT path |
| Per-operation placement | CPU kernels in `.hxb` are descriptions, not code | Memory planning |
| `.hxb` artifacts, CPU and GPU | | More operations and a frontend |

Two things worth knowing before you benchmark anything: the generated GPU
kernel launches with a single block and a single thread, so it is correct and
slow; and the JIT's GPU path passes host pointers to the device, so a
GPU-placed `-emit=jit` faults. The artifact path does not have that problem,
because the runtime's buffers are device-local by construction.

## Contributing

Fork, branch, keep `make check-hexir` green, open a pull request. New tests
should compile their own `.mlir` input and pass `-placement` explicitly rather
than relying on defaults.

## License

Apache 2.0.

## Acknowledgments

Built on [MLIR](https://mlir.llvm.org/). The dialect scaffolding started from
the MLIR Toy tutorial.

---

<div align="center"><sub>Research software under active development — APIs and IR may change.</sub></div>
