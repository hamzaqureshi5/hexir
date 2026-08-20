// The device path, starting from the kernel level. This is the lowering that
// produces the device code embedded in a .hxb: the phase and the serializer
// call the same two pipeline builders, so what this prints is what gets
// compiled.
//
// No CUDA toolkit is needed. The phase stops at NVVM, which is the IR ptxas is
// handed, so the device path can be checked on a machine with no CUDA at all.

// RUN: %hexir --compile-to=device -placement=hexir.linear=cuda %s 2>&1 | FileCheck %s

// The prim funcs stay: erasing one would leave the hexir.call_tir that names it
// dangling. The cuda one is marked as having been lowered.
// CHECK:      hextir.prim_func @linear_0
// CHECK-SAME:   device = "cuda"
// CHECK-SAME:   hexir.lowered_to_gpu

// The cpu kernel is left alone.
// CHECK:      hextir.prim_func @relu_1
// CHECK-SAME:   device = "cpu"

// The cuda kernel becomes a gpu.module carrying its launch geometry, which is
// what the serializer reads back into the executable descriptor. Two
// dimensions, because a block caps at 1024 threads.
// CHECK:      gpu.module @linear_0_module
// CHECK-SAME:   hexir.block_x = 2
// CHECK-SAME:   hexir.grid_x = 1
// CHECK-SAME:   hexir.grid_y = 2

// Bare-pointer calling convention: one device address per buffer, rather than a
// seven-scalar memref descriptor the runtime would have to build by hand.
// CHECK:      llvm.func @linear_0(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr)
// CHECK-SAME:   gpu.kernel

// Each bound axis reads its own hardware index. Both blockIdx loops reading .x
// was a real bug once, and the kernel then computed one row of the answer very
// quickly.
// CHECK:      nvvm.read.ptx.sreg.ctaid.y
// CHECK:      nvvm.read.ptx.sreg.ctaid.x
// CHECK:      nvvm.read.ptx.sreg.tid.x

func.func @main() {
  %a = hexir.constant dense<[[3.0, 1.0], [2.0, 2.0]]> : tensor<2x2xf64>
  %b = hexir.constant dense<[[1.0, 5.0], [5.0, 2.0]]> : tensor<2x2xf64>
  %m = hexir.linear %a, %b : tensor<2x2xf64>
  %r = hexir.relu %m : tensor<2x2xf64>
  hexir.print %r : tensor<2x2xf64>
  return
}
