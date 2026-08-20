# Running Hexir on the CUDA server (A6000)

All CUDA code paths are already implemented and tested up to the point where the
CUDA toolkit is required. This is the setup + verification runbook for the server.

## What is already in the repo (no changes needed)

| Component | File | Status |
|---|---|---|
| GPU partitioning (`hexir.linear` → cuda) | `src/TargetInfo.cpp`, `src/Partition.cpp` | ✅ tested locally |
| Device-attr propagation through lowering | `src/LowerToLinalg.cpp` | ✅ tested locally |
| Matmul GPU kernel (`gpu.launch`) | `src/LowerCudaToGpu.cpp::lowerMatmul` | ✅ IR verified locally |
| Generic elementwise GPU kernel (relu/add/…) | `src/LowerCudaToGpu.cpp::lowerElementwise` | ✅ IR verified locally |
| Kernel outlining → NVVM → CUBIN → runtime calls | `src/main.cpp` (pass pipeline, sm_86) | ⏳ needs CUDA toolkit |
| CUDA runtime in `hexir-run` | `runtime/src/hal/cuda/` (libcuda dlopened) | ✅ working |

## Server setup (one-time)

```bash
# 1. CUDA toolkit (provides nvcc, ptxas, libcudart)
sudo apt install nvidia-cuda-toolkit
nvcc --version   # verify

# 2. Build LLVM/MLIR
cd ~/llvm-project/build
cmake .. -DLLVM_ENABLE_PROJECTS="mlir" \
         -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
         -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install   # or point Hexir's CMakeLists at this build dir

# 3. Build hexir
cd ~/hexir && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

Notes:
- Kernels target **sm_86** (A6000), set in `src/main.cpp` (`nvvmOpts.chip`). For a
  different GPU, change that one line (e.g. `sm_80` for A100, `sm_89` for L4/4090).
- `hexir-run` dlopens `libcuda` rather than linking it, so the same binary works
  on CPU-only machines.

## Verification sequence (run in order)

```bash
cd build

# 1. CPU baseline — must print: 8.000000 17.000000 / 12.000000 14.000000
./hexir -emit=hxb -o cpu.hxb && ./hexir-run cpu.hxb

# 2. Inspect GPU IR (works even before toolchain is verified)
./hexir -emit=mlir-gpu                  # expect: gpu.launch {device="cuda"}

# 3. Full LLVM lowering with kernel compilation (first toolkit-dependent step)
./hexir -emit=mlir-llvm 2>&1 | grep -E "gpu.binary|mgpu" | head
# expect: gpu.binary blob + mgpuModuleLoad/mgpuLaunchKernel calls

# 4. Heterogeneous execution: matmul on A6000, relu on CPU  ← the goal
./hexir -emit=hxb -o m.hxb && ./hexir-run m.hxb
# expected output (identical to CPU baseline):
# 8.000000 17.000000
# 12.000000 14.000000

# 5. Everything on GPU (relu via the generic elementwise kernel)
./hexir -emit=hxb -o gpu.hxb -placement=hexir.linear=cuda,hexir.relu=cuda && ./hexir-run --device=cuda gpu.hxb

# 6. Full test suite — the 3 CUDA-gated tests un-skip automatically
#    (lit detects nvcc on PATH and enables the 'cuda' feature)
make check-hexir       # expect: 9/9 pass, 0 unsupported
```

## Placement cheat-sheet

```bash
./hexir -emit=hxb -o m.hxb                                   # everything on the CPU (default)
./hexir -emit=hxb -o cpu.hxb && ./hexir-run cpu.hxb                # all CPU
./hexir -emit=hxb -o gpu.hxb -placement=hexir.linear=cuda,hexir.relu=cuda && ./hexir-run --device=cuda gpu.hxb                  # all GPU
./hexir -emit=hxb -o cpu.hxb && ./hexir-run cpu.hxb,hexir.relu=gpu # swapped
```

Defaults live in `src/TargetInfo.cpp` (`opPreferred_`); `gpu` ≡ `cuda`.

## If something fails

| Symptom | Likely cause |
|---|---|
| `error: ... gpu-module-to-binary` / "Failed to compile" | ptxas not on PATH → check `which ptxas` |

| `CUDA_ERROR_NO_BINARY_FOR_GPU` | chip mismatch → set `nvvmOpts.chip` to your GPU's sm_XX |
| Wrong numeric results on GPU | report back — likely a missing host↔device transfer (known roadmap item) |
