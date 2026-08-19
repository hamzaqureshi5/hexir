// The -placement flag reroutes any op to cpu or cuda at runtime, keyed on
// the frontend hexir op names ("gpu" is accepted as an alias for "cuda").

// Default: linear (matmul) on GPU, relu on CPU.
// RUN: %hexir -emit=mlir-hetero 2>&1 | FileCheck %s --check-prefix=DEFAULT
// DEFAULT: ls_gpu.matmul
// DEFAULT: ls_cpu.relu

// Everything on CPU.
// RUN: %hexir -emit=mlir-hetero -placement=hexir.linear=cpu 2>&1 | FileCheck %s --check-prefix=ALLCPU
// ALLCPU: ls_cpu.matmul
// ALLCPU: ls_cpu.relu

// Swapped: linear on CPU, relu on GPU (using the "gpu" alias).
// RUN: %hexir -emit=mlir-hetero -placement=hexir.linear=cpu,hexir.relu=gpu 2>&1 | FileCheck %s --check-prefix=SWAP
// SWAP: ls_cpu.matmul
// SWAP: ls_gpu.relu

// Everything on GPU — relu lowers via the generic elementwise GPU kernel.
// RUN: %hexir -emit=mlir-gpu -placement=hexir.relu=cuda 2>&1 | FileCheck %s --check-prefix=ALLGPU
// ALLGPU: gpu.launch
// ALLGPU: arith.mulf
// ALLGPU: gpu.launch
// ALLGPU: arith.maximumf

// Invalid device is rejected.
// RUN: not %hexir -emit=mlir-hetero -placement=hexir.linear=tpu 2>&1 | FileCheck %s --check-prefix=BADDEV
// BADDEV: invalid -placement entry
