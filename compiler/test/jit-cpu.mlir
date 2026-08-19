// End-to-end on CPU only: -placement reroutes the linear (matmul) off the
// GPU, so this runs on any machine without the CUDA toolkit.
// RUN: %hexir -emit=jit -placement=hexir.linear=cpu 2>&1 | FileCheck %s
// RUN: %hexir -emit=jit -placement=hexir.linear=cpu -opt 2>&1 | FileCheck %s

// CHECK:      8.000000 17.000000
// CHECK-NEXT: 12.000000 14.000000
