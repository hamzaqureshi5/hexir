// LowerToTIRPass: each hexir compute op becomes a hextir.prim_func plus a
// hexir.call_tir. Placement picks the loop kinds -- cpu gets "parallel",
// cuda gets "thread_binding" with a bound axis. Reductions stay "serial".
//
// RUN: %hexir -emit=mlir-tir %s 2>&1 | FileCheck %s --check-prefix=CPU
// RUN: %hexir -emit=mlir-tir -placement=hexir.relu=cuda %s 2>&1 | FileCheck %s --check-prefix=SPLIT

func.func @main() {
  %a = hexir.constant dense<[[3.0, 1.0], [2.0, 2.0]]> : tensor<2x2xf64>
  %b = hexir.constant dense<[[1.0, 5.0], [5.0, 2.0]]> : tensor<2x2xf64>
  %m = hexir.linear %a, %b : tensor<2x2xf64>
  %r = hexir.relu %m : tensor<2x2xf64>
  hexir.print %r : tensor<2x2xf64>
  return
}

// All on the host, so both parallel axes are "parallel" and k stays "serial".
// CPU:       hexir.call_tir @linear_0(%{{.*}}, %{{.*}}) {device = "cpu"}
// CPU:       hexir.call_tir @relu_1(%{{.*}}) {device = "cpu"}
// CPU-NOT:   hexir.linear
// CPU-NOT:   hexir.relu
// CPU:       hextir.prim_func @linear_0(%{{.*}}: memref<2x2xf64>, %{{.*}}: memref<2x2xf64>, %{{.*}}: memref<2x2xf64>)
// CPU-SAME:    device = "cpu"
// CPU:         hextir.block "matmul"
// CPU:           hextir.for "parallel"
// CPU:             hextir.for "parallel"
// CPU:               hextir.for "serial"
// CPU:                 arith.mulf
// CPU:                 arith.addf
// CPU:       hextir.prim_func @relu_1(%{{.*}}: memref<2x2xf64>, %{{.*}}: memref<2x2xf64>)
// CPU:         arith.maximumf

// Split placement -- the matmul stays on the host, the relu is bound to GPU
// axes, and both kernels sit in the same module.
// SPLIT:      hexir.call_tir @linear_0(%{{.*}}, %{{.*}}) {device = "cpu"}
// SPLIT:      hexir.call_tir @relu_1(%{{.*}}) {device = "cuda"}
// SPLIT:      hextir.prim_func @linear_0
// SPLIT-SAME:   device = "cpu"
// SPLIT:        hextir.for "parallel"
// SPLIT:      hextir.prim_func @relu_1
// SPLIT-SAME:   device = "cuda"
// SPLIT:        hextir.for "thread_binding" %{{.*}} to %{{.*}} step %{{.*}} bind "blockIdx.x"
// SPLIT:          hextir.for "thread_binding" %{{.*}} to %{{.*}} step %{{.*}} bind "threadIdx.x"
// SPLIT:            arith.maximumf
