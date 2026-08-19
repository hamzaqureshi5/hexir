// End to end through the artifact: compile to a loadable module, then run it
// with hexir-run, which links no MLIR and no LLVM. The numbers must match what
// the JIT produces for the same program.
//
// REQUIRES: runtime, compiler
// RUN: %hexir -emit=hxb -o %t.hxb %s
// RUN: %hexir-run %t.hxb 2>&1 | FileCheck %s
// RUN: not %hexir-run %t.hxb --entry=nosuchfunc 2>&1 | FileCheck %s --check-prefix=MISSING

func.func @main() {
  %a = hexir.constant dense<[[3.0, 1.0], [2.0, 2.0]]> : tensor<2x2xf64>
  %b = hexir.constant dense<[[1.0, 5.0], [5.0, 2.0]]> : tensor<2x2xf64>
  %m = hexir.linear %a, %b : tensor<2x2xf64>
  %s = hexir.add %m, %b : tensor<2x2xf64>
  %r = hexir.relu %s : tensor<2x2xf64>
  hexir.print %r : tensor<2x2xf64>
  return
}

// Four sections, one executable per compute op.
// CHECK:      sections      : 4
// CHECK:      symbols
// CHECK:      program
// CHECK:      rodata
// CHECK:      executables
// CHECK:      device        : cpu

// [[3,1],[2,2]] x [[1,5],[5,2]] = [[8,17],[12,14]], plus b, then relu.
// CHECK:      9.000000 22.000000
// CHECK-NEXT: 17.000000 16.000000

// An entry point that is not in the symbol table is reported, not crashed on.
// MISSING: not found
