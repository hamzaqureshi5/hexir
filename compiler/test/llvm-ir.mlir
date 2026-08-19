// Translation to LLVM IR: host code with CUDA runtime calls + CPU relu loops.
// Requires the CUDA toolkit (nvcc/ptxas) to compile the matmul kernel.
// REQUIRES: cuda
// RUN: %hexir -emit=llvm 2>&1 | FileCheck %s

// CHECK: target triple
// CHECK: declare {{.*}}i32 @printf(ptr, ...)
// CHECK: define {{.*}}void @main()
// CHECK: call i32 (ptr, ...) @printf
