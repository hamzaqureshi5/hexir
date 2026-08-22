# Getting started

## Build

Hexir needs an **unreleased** LLVM — it uses the new-style
`OpTy::create(builder, ...)` MLIR builders, so no packaged MLIR will do. The
commit it works with is pinned as a submodule, and one script builds it:

```bash
git clone --recurse-submodules git@github.com:hamzaqureshi5/hexir.git && cd hexir
scripts/build-llvm.sh                 # 30-90 min, ~10 GB, once
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

The script turns on the CUDA pieces only when `nvcc` is on PATH, so this works
on a machine with no GPU; the CPU path is complete either way.

If you already have an LLVM 24 build with MLIR, point at it and skip the
submodule entirely:

```bash
cmake -S . -B build -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
                    -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir
```

The configure prints which LLVM it picked. That matters more than it sounds:
pointing `LLVM_DIR` at a directory with no config in it does not fail, it makes
CMake search its default prefixes instead.

This produces two programs:

`build/hexir`
: The compiler. Large, because it links MLIR and LLVM.

`build/hexir-run`
: The runtime. Small, because it links neither.

## Run something

With no input file, the compiler builds a small program in C++ (a 2x2 matrix
multiply) and compiles that. It is the quickest way to check your build works:

```bash
./build/hexir --compile-to=artifact -o hello.hxb
./build/hexir-run hello.hxb
```

```text
8.000000 17.000000
12.000000 14.000000
```

## Compile your own program

Write the input in the `hexir` dialect:

```mlir
func.func @main() {
  %a = hexir.constant dense<[[3.0, -1.0], [2.0, 2.0]]> : tensor<2x2xf64>
  %b = hexir.constant dense<[[1.0, 5.0], [5.0, -2.0]]> : tensor<2x2xf64>
  %m = hexir.linear %a, %b : tensor<2x2xf64>
  %r = hexir.relu %m : tensor<2x2xf64>
  hexir.print %r : tensor<2x2xf64>
  return
}
```

Then:

```bash
./build/hexir --compile-to=artifact -o mine.hxb mine.mlir
./build/hexir-run mine.hxb
```

Five operations are supported end to end: `constant`, `linear`, `add`, `relu`
and `print`. Others are declared in the dialect but have no lowering yet.

## Look inside

Every stage of the pipeline can be printed. This is the most useful thing about
Hexir for learning:

```bash
./build/hexir --compile-to=input mine.mlir    # the graph you wrote
./build/hexir --compile-to=kernel mine.mlir   # each op as a kernel
./build/hexir --compile-to=place mine.mlir    # where each op runs
./build/hexir --compile-to=device mine.mlir   # GPU kernels, as NVVM
./build/hexir --compile-to=artifact -o mine.hxb mine.mlir   # a module
```

## Choose where things run

By default everything runs on the CPU. Move an operation with `-placement`:

```bash
./build/hexir --compile-to=kernel -placement=hexir.linear=cuda mine.mlir
```

Look at the loops in the output. On the CPU they are `parallel`; on CUDA they
are `thread_binding` and bound to a GPU axis.

You can move more than one at a time:

```bash
./build/hexir --compile-to=kernel -placement=hexir.linear=cuda,hexir.relu=cpu mine.mlir
```

## Run it on the GPU

With a CUDA toolkit installed and LLVM built for NVPTX, a GPU-placed kernel
compiles to a CUBIN that gets embedded in the module:

```bash
./build/hexir --compile-to=artifact -o gpu.hxb -placement=hexir.linear=cuda mine.mlir
./build/hexir-run --device=cuda gpu.hxb
```

```text
device        : cuda (NVIDIA GeForce GTX 1660 Ti)
--
8.000000 17.000000
12.000000 14.000000
```

No environment setup is needed: Hexir finds `libdevice` itself, including on
distributions that do not use NVIDIA's directory layout. Build instructions for
LLVM are in `docs/llvm-cuda-build.txt`, and the setup runbook is
[Running on a CUDA server](cuda-server.md).

:::{note}
A module whose kernels are split across devices cannot run yet: the runtime
opens one device and runs the whole program on it. Place every compute op on
the same device, or run the CPU and GPU versions separately.
:::

## Ship a file instead

```bash
./build/hexir --compile-to=artifact -o model.hxb mine.mlir
./build/hexir-run model.hxb
```

`hexir-run` contains no compiler. This is the deploy path — see
[The runtime](runtime.md).

## Run the tests

The suite needs `lit` and `FileCheck`:

```bash
pip install lit
cd build && make check-hexir             # both suites
cd build && make check-hexir-runtime     # just the runtime
```

All 15 pass. Three need a CUDA toolkit and report as unsupported without one.
