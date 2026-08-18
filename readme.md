<div align="center">
  <img src="assets/logo.svg" width="140" alt="Hexir logo"/>

  # Hexir
    
  **Hexir (Heterogeneous EXecution IR) is a research MLIR compiler**

  *One graph in — partitioned, lowered, and executed across CPU and GPU.*

  [![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
  [![MLIR](https://img.shields.io/badge/LLVM-MLIR-orange.svg)](https://mlir.llvm.org/)
  [![CMake](https://img.shields.io/badge/CMake-3.13.4%2B-064F8C.svg)](https://cmake.org/)
  [![Tests](https://img.shields.io/badge/tests-lit%20%2B%20FileCheck-brightgreen.svg)](#testing)
  [![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

Hexir (**H**eterogeneous **EX**ecution **IR**) is a research compiler that demonstrates end-to-end compilation of neural-network programs through a custom MLIR dialect: progressive lowering, automatic CPU/CUDA partitioning, and JIT execution.

```mlir
%2 = hexir.linear %0, %1 : tensor<2x2xf64>   // placed on GPU as linalg.matmul
%3 = hexir.relu %2 : tensor<2x2xf64>         // placed on CPU
hexir.print %3 : tensor<2x2xf64>
```

## Highlights

- **Custom MLIR dialect** — `hexir` ops for neural-network primitives (`linear`, `relu`, `sigmoid`, `softmax`, …), defined in TableGen with shape inference
- **Progressive lowering** — `hexir` → linalg/tensor → bufferized memref → SCF/CF → LLVM IR, inspectable at every stage
- **Heterogeneous partitioning** — a partition pass assigns each op a device based on a target-support registry; compute-heavy ops go to CUDA, element-wise and I/O stay on CPU
- **GPU dialect generation** — CUDA-partitioned ops lower to `gpu.launch` kernels
- **JIT execution** — the CPU path compiles and runs in-process via MLIR's ExecutionEngine
- **Lit/FileCheck test suite** — one test per pipeline stage

## Architecture

```
                         hexir dialect       -->    (high-level NN ops, shape inference)
                              │
                              ▼
                    linalg / arith / tensor  -->    (structured ops on tensors)
                              │
                              ▼
                         partitioning pass   -->    (device = "cpu" | "cuda" per op)
                         ┌─────┴─────┐
                         ▼           ▼
                    CPU partition   CUDA partition
                         │           │
                         ▼           ▼
                    bufferization   gpu.launch kernels
                         │
                         ▼
                    SCF → CF → LLVM dialect
                         │
                         ▼
                    LLVM IR → JIT execution
```


### Build

```bash
git clone git@github.com:hamzaqureshi5/hexir.git && cd hexir
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

Every stage of the pipeline can be dumped for inspection:

| Emit mode | Output |
|---|---|
| `-emit=mlir` | Initial module in the `hexir` dialect |
| `-emit=mlir-linalg` | After lowering to `linalg`/`arith`/`tensor` |
| `-emit=mlir-hetero` | After CPU/CUDA partitioning (`ls_cpu`/`ls_gpu` ops) |
| `-emit=mlir-gpu` | CUDA partitions as `gpu.launch` kernels |
| `-emit=mlir-llvm` | CPU path in the LLVM dialect |
| `-emit=llvm` | Translated LLVM IR |
| `-emit=jit` | JIT-compile and execute |

Add `-opt` to enable optimizations, or `--print-ir-after-all` to trace the pass pipeline:

```bash
./hexir -emit=mlir-hetero
./hexir -emit=jit -opt
```

### Example: partitioned IR

```mlir
module attributes {hexir.targets = ["cpu", "cuda"]} {
  %0 = ls_gpu.matmul %cst, %cst_0 : tensor<2x2xf64>   // GPU
  %1 = ls_cpu.relu %0 : tensor<2x2xf64>               // CPU
  hexir.print %1 {device = "cpu"} : memref<2x2xf64>
}
```

> **Note** — the CUDA path currently generates MLIR GPU IR for inspection; only the CPU path executes. Full CUDA runtime integration is on the roadmap.

## Testing

The lit/FileCheck suite covers each pipeline stage plus end-to-end JIT execution:

```bash
cd build && make check-hexir     # full suite
lit -v test                      # from the repo root
lit -v test --filter=jit         # single test
```

## Roadmap

- [ ] CUDA runtime integration (kernel launch, host↔device transfers)
- [ ] Graph-level partitioning (clustered subgraphs instead of per-op placement)
- [ ] Memory-transfer insertion at partition boundaries
- [ ] Additional NN operations and frontends
- [ ] Additional backends (Metal, ROCm, RISC-V scaffolding in `targets/`)

## Contributing

1. Fork the repository and create a feature branch
2. Make your changes; keep `make check-hexir` green
3. Open a Pull Request

## License

Apache License 2.0.

## Acknowledgments

Built on the [LLVM MLIR](https://mlir.llvm.org/) infrastructure; the dialect scaffolding draws on the MLIR Toy tutorial. Part of ongoing research in heterogeneous compilation for machine learning.

---

<div align="center"><sub>Hexir is research software under active development — APIs and IR may change without notice.</sub></div>
