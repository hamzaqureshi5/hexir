// How far to compile and what to write out are two separate flags. Each phase
// stops where it says it does, and a pair that cannot mean anything is refused
// rather than guessed at.

// RUN: %hexir --compile-to=input %s 2>&1 | FileCheck %s --check-prefix=INPUT
// RUN: %hexir --compile-to=graph %s 2>&1 | FileCheck %s --check-prefix=GRAPH
// RUN: %hexir --compile-to=place %s 2>&1 | FileCheck %s --check-prefix=PLACE
// RUN: %hexir --compile-to=kernel %s 2>&1 | FileCheck %s --check-prefix=KERNEL
// RUN: %hexir --compile-to=linalg %s 2>&1 | FileCheck %s --check-prefix=LINALG

// The graph as written, no passes and no placement.
// INPUT:       hexir.linear
// INPUT-NOT:   device
// INPUT-NOT:   linalg

// Graph optimization does not change this program, but it is its own phase.
// GRAPH:       hexir.linear

// Placement, still on the frontend ops.
// PLACE:       module attributes {hexir.targets = ["cpu", "cuda"]}
// PLACE:       hexir.linear %{{.*}} {device = "cpu"}
// PLACE-NOT:   linalg

// The kernel level.
// KERNEL:      hexir.call_tir @linear_0
// KERNEL:      hextir.prim_func @linear_0
// KERNEL:      hextir.for "parallel"

// The frozen route, which does run placement first.
// LINALG:      linalg.matmul {device = "cpu"}
// LINALG-NOT:  hexir.linear

// An artifact is a file, and MLIR is not.
// RUN: not %hexir --compile-to=artifact -emit=mlir %s 2>&1 | FileCheck %s --check-prefix=NOTMLIR
// NOTMLIR: writes a file, not MLIR

// hxb is only produced by the artifact phase.
// RUN: not %hexir --compile-to=kernel -emit=hxb %s 2>&1 | FileCheck %s --check-prefix=NOTHXB
// NOTHXB: only makes sense with --compile-to=artifact

// No phase but linalg-llvm reaches LLVM IR yet, and saying so beats emitting
// the wrong thing.
// RUN: not %hexir --compile-to=kernel -emit=llvm %s 2>&1 | FileCheck %s --check-prefix=NOTLLVM
// NOTLLVM: no other phase reaches LLVM IR yet

// A format that does not exist names the ones that do.
// RUN: not %hexir -emit=bogus %s 2>&1 | FileCheck %s --check-prefix=BADFMT
// BADFMT: is not an output format

// The old spellings still work, and say what to use instead.
// RUN: %hexir -emit=mlir-tir %s 2>&1 | FileCheck %s --check-prefix=OLDNAME
// OLDNAME: deprecated; use --compile-to=kernel
// OLDNAME: hextir.prim_func

// An old spelling that disagrees with an explicit phase is a mistake, not a
// precedence puzzle.
// RUN: not %hexir -emit=mlir-tir --compile-to=linalg %s 2>&1 | FileCheck %s --check-prefix=CLASH
// CLASH: ask for different phases

func.func @main() {
  %a = hexir.constant dense<[[3.0, 1.0], [2.0, 2.0]]> : tensor<2x2xf64>
  %b = hexir.constant dense<[[1.0, 5.0], [5.0, 2.0]]> : tensor<2x2xf64>
  %m = hexir.linear %a, %b : tensor<2x2xf64>
  hexir.print %m : tensor<2x2xf64>
  return
}
