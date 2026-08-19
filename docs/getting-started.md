# Getting started

## Build

Hexir needs a build of LLVM with MLIR. Point CMake at it:

```bash
git clone git@github.com:hamzaqureshi5/hexir.git && cd hexir
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
         -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
make -j$(nproc)
```

This produces two programs:

`build/hexir`
: The compiler. Large, because it links MLIR and LLVM.

`build/hexir-run`
: The runtime. Small, because it links neither.

## Run something

With no input file, the compiler builds a small program in C++ (a 2x2 matrix
multiply) and compiles that. It is the quickest way to check your build works:

```bash
./build/hexir -emit=jit
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
./build/hexir -emit=jit mine.mlir
```

Five operations are supported end to end: `constant`, `linear`, `add`, `relu`
and `print`. Others are declared in the dialect but have no lowering yet.

## Look inside

Every stage of the pipeline can be printed. This is the most useful thing about
Hexir for learning:

```bash
./build/hexir -emit=mlir mine.mlir           # the graph you wrote
./build/hexir -emit=mlir-tir mine.mlir       # each op as a kernel
./build/hexir -emit=mlir-linalg mine.mlir    # loops on tensors
./build/hexir -emit=mlir-gpu mine.mlir       # GPU kernels
./build/hexir -emit=llvm mine.mlir           # LLVM IR
```

## Choose where things run

By default everything runs on the CPU. Move an operation with `-placement`:

```bash
./build/hexir -emit=mlir-tir -placement=hexir.linear=cuda mine.mlir
```

Look at the loops in the output. On the CPU they are `parallel`; on CUDA they
are `thread_binding` and bound to a GPU axis.

You can move more than one at a time:

```bash
./build/hexir -emit=mlir-tir -placement=hexir.linear=cuda,hexir.relu=cpu mine.mlir
```

## Run it on the GPU

With a CUDA toolkit installed and LLVM built for NVPTX, a GPU-placed kernel
compiles to a CUBIN that gets embedded in the module:

```bash
./build/hexir -emit=hxb -o gpu.hxb -placement=hexir.linear=cuda mine.mlir
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

:::{warning}
`-emit=jit` with a CUDA placement currently faults. The generated host code
passes host pointers to the device with no transfers. The artifact path above
does not have this problem, because the runtime allocates device-local buffers
and copies into them.
:::

## Ship a file instead

```bash
./build/hexir -emit=hxb -o model.hxb mine.mlir
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
