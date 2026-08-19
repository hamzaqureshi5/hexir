// End-to-end: matmul on CUDA A6000 (sm_86), relu on CPU.
// Requires the CUDA toolkit and libmlir_cuda_runtime.so.
// REQUIRES: cuda
// RUN: %hexir -emit=jit 2>&1 | FileCheck %s
// RUN: %hexir -emit=jit -opt 2>&1 | FileCheck %s

// CHECK:      8.000000 17.000000
// CHECK-NEXT: 12.000000 14.000000
