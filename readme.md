<div align="center">
  <img src="assets/logo.svg" width="140" alt="Hexir logo"/>

  # Hexir

  **Heterogeneous MLIR Compiler for Neural Networks**

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

Hexir (**H**eterogeneous **EX**ecution **IR**) is an MLIR compiler with two
custom dialects: a graph level of neural-network operators on tensors, and a
kernel level where each loop carries its own schedule, so a scheduling decision
is written in the IR instead of being implied by which pass ran.

A partitioning pass assigns each operator to CPU or CUDA before lowering and
propagates that placement through every conversion — linalg → bufferization →
SCF → LLVM IR on the host, GPU dialect → NVVM → CUBIN on the device.

The output is a self-describing artifact with the device code embedded, run by a
standalone C99 runtime that links no MLIR or LLVM. On a GTX 1660 Ti a GPU matmul
reaches 47–68 GFLOP/s in f64 against cuBLAS at 123–179, checked against a CPU
reference.

It is small enough to read. That is deliberate: the gap between MLIR's Toy
tutorial and a production compiler like IREE is enormous, and Hexir sits in the
middle — a complete pipeline, in about five thousand lines.

```mlir
%m = hexir.linear %a, %w : tensor<2x2xf64>   // placed on the GPU
%r = hexir.relu %m : tensor<2x2xf64>         // placed on the CPU
hexir.print %r : tensor<2x2xf64>
```

## Compile, then run

```bash
hexir -emit=hxb -o model.hxb                 # compile to a file
hexir-run model.hxb                          # run it later, no compiler present
```

There is one way to run a program: compile it to a file, then run the file.
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
| CPU and GPU, end to end | GPU kernels have no tiling yet | Multi-device modules in one run |
| Per-operation placement | CPU kernels in `.hxb` are descriptions, not code | f32; memory planning |
| `.hxb` artifacts, CPU and GPU | | More operations and a frontend |

Benchmarks live in `bench/`. On a GTX 1660 Ti a matmul currently runs about
2.6x slower than cuBLAS in f64, verified against a CPU reference. The kernel
has no shared-memory tiling yet, and the whole dialect is f64 on a card whose
f64 rate is 1/32 of its f32 rate, so there is a lot of headroom in both.

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
