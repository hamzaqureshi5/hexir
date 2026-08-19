// LowerToLLVMPass: full pipeline including GPU kernel compilation.
// Requires the CUDA toolkit (nvcc/ptxas) to compile the matmul kernel.
// REQUIRES: cuda
// RUN: %hexir -emit=mlir-llvm 2>&1 | FileCheck %s

// Host side: CUDA runtime calls for matmul launch, CPU loops for relu.
// CHECK:       llvm.func @printf(!llvm.ptr, ...)
// CHECK:       llvm.func @main()
// CHECK-NOT:   linalg.
// CHECK-NOT:   scf.
// CHECK-NOT:   hexir.
