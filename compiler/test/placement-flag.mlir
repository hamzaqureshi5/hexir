// The -placement flag reroutes any op to cpu or cuda without recompiling. Keys
// are frontend hexir op names, because partitioning runs before lowering.
// "gpu" is accepted as an alias for "cuda".

// Everything on the CPU. This is the default, so no flag is needed.
// RUN: %hexir -emit=mlir-hetero %s 2>&1 | FileCheck %s --check-prefix=ALLCPU
// ALLCPU: linalg.matmul {device = "cpu"}
// ALLCPU: linalg.generic
// ALLCPU-SAME: device = "cpu"

// Matmul moved to the GPU, relu left behind.
// RUN: %hexir -emit=mlir-hetero -placement=hexir.linear=cuda %s 2>&1 | FileCheck %s --check-prefix=SPLIT
// SPLIT: linalg.matmul {device = "cuda"}
// SPLIT: linalg.generic
// SPLIT-SAME: device = "cpu"

// The other way round, using the "gpu" alias, and both flags at once.
// RUN: %hexir -emit=mlir-hetero -placement=hexir.linear=cpu,hexir.relu=gpu %s 2>&1 | FileCheck %s --check-prefix=SWAP
// SWAP: linalg.matmul {device = "cpu"}
// SWAP: linalg.generic
// SWAP-SAME: device = "cuda"

// Both on the GPU: each becomes its own gpu.launch, the matmul with a multiply
// and the relu with a maximumf.
// RUN: %hexir -emit=mlir-gpu -placement=hexir.linear=cuda,hexir.relu=cuda %s 2>&1 | FileCheck %s --check-prefix=ALLGPU
// ALLGPU: gpu.launch
// ALLGPU: arith.mulf
// ALLGPU: gpu.launch
// ALLGPU: arith.maximumf

// A device the op does not support is rejected before any pass runs.
// RUN: not %hexir -emit=mlir-hetero -placement=hexir.linear=tpu %s 2>&1 | FileCheck %s --check-prefix=BADDEV
// BADDEV: invalid -placement entry

func.func @main() {
  %a = hexir.constant dense<[[3.0, 1.0], [2.0, 2.0]]> : tensor<2x2xf64>
  %b = hexir.constant dense<[[1.0, 5.0], [5.0, 2.0]]> : tensor<2x2xf64>
  %m = hexir.linear %a, %b : tensor<2x2xf64>
  %r = hexir.relu %m : tensor<2x2xf64>
  hexir.print %r : tensor<2x2xf64>
  return
}
