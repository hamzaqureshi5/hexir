// CudaGpuLoweringPass: a matmul placed on cuda becomes a gpu.launch kernel
// after bufferization, while a relu left on the CPU stays as linalg.generic.
// Both live in one module -- that is the heterogeneous case.
//
// This only checks the IR, so it needs no CUDA toolkit. Compiling the kernel
// to CUBIN does, and that is covered by the cuda-gated tests instead.
//
// RUN: %hexir -emit=mlir-gpu -placement=hexir.linear=cuda %s 2>&1 | FileCheck %s

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

// The matmul became a GPU kernel, and the launch keeps the placement that
// produced it.
// CHECK:         gpu.launch blocks
// CHECK:           memref.load
// CHECK:           arith.mulf
// CHECK:           arith.addf
// CHECK:           memref.store
// CHECK:           gpu.terminator
// CHECK:         } {device = "cuda"}

// The relu stayed behind on the host.
// CHECK:         linalg.generic
// CHECK-SAME:      device = "cpu"
// CHECK:           arith.maximumf
// CHECK:         hexir.print %{{.*}} {device = "cpu"}
