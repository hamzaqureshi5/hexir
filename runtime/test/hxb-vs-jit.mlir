// The artifact path and the JIT must agree.
//
// This is the differential test: compile the same program twice, once to a
// module executed by a runtime that links no MLIR, once through the in-process
// JIT, and require the output to be identical. Placement, serialization and
// the command list all have to be right for this to hold -- a wrong slot
// assignment or a swapped matmul operand shows up here as a diff.
//
// REQUIRES: runtime, compiler
// RUN: %hexir -emit=jit %s > %t.jit
// RUN: %hexir -emit=hxb -o %t.hxb %s
// RUN: %hexir-run --quiet %t.hxb > %t.rt
// RUN: diff %t.jit %t.rt
//
// And the numbers are actually what the arithmetic says, not just consistent.
// RUN: %hexir-run --quiet %t.hxb | FileCheck %s

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
