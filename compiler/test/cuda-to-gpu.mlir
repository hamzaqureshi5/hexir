// CudaGpuLoweringPass: cuda-partitioned matmul becomes a gpu.launch kernel
// (after bufferization), while the relu stays on the CPU as linalg.generic.
// RUN: %hexir -emit=mlir-gpu 2>&1 | FileCheck %s

// CHECK:       module attributes {hexir.targets = ["cpu", "cuda"]}
// CHECK-LABEL: func.func @main()
// CHECK:         gpu.launch blocks
// CHECK:           memref.load
// CHECK:           arith.mulf
// CHECK:           arith.addf
// CHECK:           memref.store
// CHECK:           gpu.terminator
// CHECK:         } {device = "cuda"}
// CHECK:         linalg.generic
// CHECK-SAME:      device = "cpu"
// CHECK:           arith.maximumf
// CHECK:         hexir.print %{{.*}} {device = "cpu"}
