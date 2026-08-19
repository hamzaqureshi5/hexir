// The full artifact path on a GPU: compile to a module that carries a real
// device image, then execute it with a runtime that links no MLIR and no LLVM.
//
// The numbers must match the CPU, which is the point -- the same program, the
// same answer, a different device, and no compiler in the process.
//
// REQUIRES: runtime, compiler, cuda
// RUN: %hexir -emit=hxb -o %t.gpu.hxb -placement=hexir.linear=cuda %s
// RUN: %hexir -emit=hxb -o %t.cpu.hxb %s
// RUN: %hexir-run --quiet --device=cuda %t.gpu.hxb > %t.gpu.out
// RUN: %hexir-run --quiet %t.cpu.hxb > %t.cpu.out
// RUN: diff %t.cpu.out %t.gpu.out
// RUN: %hexir-run --quiet --device=cuda %t.gpu.hxb | FileCheck %s
//
// The module built for cuda is refused on a cpu device: there is no host
// fallback for a kernel that only exists as device code.
// RUN: not %hexir-run --quiet %t.gpu.hxb 2>&1 | FileCheck %s --check-prefix=WRONGDEV

func.func @main() {
  %a = hexir.constant dense<[[3.0, 1.0], [2.0, 2.0]]> : tensor<2x2xf64>
  %b = hexir.constant dense<[[1.0, 5.0], [5.0, 2.0]]> : tensor<2x2xf64>
  %m = hexir.linear %a, %b : tensor<2x2xf64>
  hexir.print %m : tensor<2x2xf64>
  return
}

// CHECK:      8.000000 17.000000
// CHECK-NEXT: 12.000000 14.000000

// WRONGDEV: placed on cuda but the active device is cpu
