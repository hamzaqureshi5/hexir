// PartitionPass: the module is tagged with the targets it was built for, and
// every op carries the device it was placed on.
//
// Placement shows up as a `device` attribute on the real linalg op. There used
// to be a detour through mirror ls_cpu/ls_gpu ops so the placement appeared in
// the op *name*; printing the actual op says more, and the round trip through
// the mirror dialects was lossy.
//
// RUN: %hexir -emit=mlir-hetero -placement=hexir.linear=cuda %s 2>&1 | FileCheck %s

func.func @main() {
  %a = hexir.constant dense<[[3.0, 1.0], [2.0, 2.0]]> : tensor<2x2xf64>
  %b = hexir.constant dense<[[1.0, 5.0], [5.0, 2.0]]> : tensor<2x2xf64>
  %m = hexir.linear %a, %b : tensor<2x2xf64>
  %r = hexir.relu %m : tensor<2x2xf64>
  hexir.print %r : tensor<2x2xf64>
  return
}

// CHECK:       module attributes {hexir.targets = ["cpu", "cuda"]}
// CHECK-LABEL: func.func @main()

// The matmul went to the GPU ...
// CHECK:         linalg.matmul {device = "cuda"}

// ... and the relu stayed on the CPU, as a generic with a maximumf body.
// CHECK:         linalg.generic
// CHECK-SAME:      device = "cpu"
// CHECK:           arith.maximumf

// I/O is never placed anywhere but the host.
// CHECK:         hexir.print %{{.*}} {device = "cpu"}

// Nothing survives that should have been lowered.
// CHECK-NOT:     hexir.linear
// CHECK-NOT:     hexir.relu
