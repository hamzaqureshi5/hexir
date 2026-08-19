// The same program compiled for each device must give the same answer.
//
// This replaces the old artifact-versus-JIT comparison. With the JIT gone the
// meaningful differential is between the two devices: placement, kernel
// generation and the launch geometry all have to be right for these to agree.
//
// REQUIRES: runtime, compiler, cuda
// RUN: %hexir -emit=hxb -o %t.cpu.hxb %s
// Both compute ops go to the GPU: the runtime opens one device and runs the
// whole program on it, so a module whose kernels are split across devices
// cannot execute yet. That is a real gap, not a property of this test.
// RUN: %hexir -emit=hxb -o %t.gpu.hxb -placement=hexir.linear=cuda,hexir.relu=cuda %s
// RUN: %hexir-run --quiet %t.cpu.hxb > %t.cpu.out
// RUN: %hexir-run --quiet --device=cuda %t.gpu.hxb > %t.gpu.out
// RUN: diff %t.cpu.out %t.gpu.out
// RUN: %hexir-run --quiet %t.cpu.hxb | FileCheck %s

func.func @main() {
  %a = hexir.constant dense<[[3.0, -1.0], [2.0, 2.0]]> : tensor<2x2xf64>
  %b = hexir.constant dense<[[1.0, 5.0], [5.0, -2.0]]> : tensor<2x2xf64>
  %m = hexir.linear %a, %b : tensor<2x2xf64>
  %r = hexir.relu %m : tensor<2x2xf64>
  hexir.print %r : tensor<2x2xf64>
  return
}

// [[3,-1],[2,2]] x [[1,5],[5,-2]] = [[-2,17],[12,6]], and relu clamps the -2.
// CHECK:      0.000000 17.000000
// CHECK-NEXT: 12.000000 6.000000
