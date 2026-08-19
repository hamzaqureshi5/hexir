// The hextir (kernel-level) dialect parses and prints, and hexir.call_tir
// bridges the graph level to it.
//
// Unlike the emit-stage tests, this one passes %s as an input file -- the
// driver compiles the file when one is given, and falls back to the program
// built in Builder.cpp when it is not.
//
// RUN: %hexir -emit=mlir %s 2>&1 | FileCheck %s

module {
  func.func @main() {
    %a = hexir.constant dense<[[3.0, 1.0], [2.0, 2.0]]> : tensor<2x2xf64>
    %b = hexir.constant dense<[[1.0, 5.0], [5.0, 2.0]]> : tensor<2x2xf64>
    %r = hexir.call_tir @add(%a, %b) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    hexir.print %r : tensor<2x2xf64>
    return
  }

  // Destination-passing: one buffer per input, plus one for the result.
  hextir.prim_func @add(%A: memref<2x2xf64>, %B: memref<2x2xf64>, %C: memref<2x2xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    hextir.block "C" {
      hextir.for "parallel" %c0 to %c2 step %c1 {
      ^bb0(%i: index):
        hextir.for "thread_binding" %c0 to %c2 step %c1 bind "threadIdx.x" {
        ^bb0(%j: index):
          %x = hextir.buffer_load %A[%i, %j] : memref<2x2xf64> -> f64
          %y = hextir.buffer_load %B[%i, %j] : memref<2x2xf64> -> f64
          %s = arith.addf %x, %y : f64
          hextir.buffer_store %s, %C[%i, %j] : f64, memref<2x2xf64>
          hextir.yield
        }
        hextir.yield
      }
      hextir.yield
    }
    hextir.return
  }
}

// CHECK-LABEL: func.func @main()
// CHECK:         %[[R:.*]] = hexir.call_tir @add(%{{.*}}, %{{.*}}) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
// CHECK:         hexir.print %[[R]] : tensor<2x2xf64>

// CHECK:       hextir.prim_func @add(%{{.*}}: memref<2x2xf64>, %{{.*}}: memref<2x2xf64>, %{{.*}}: memref<2x2xf64>)
// CHECK:         hextir.block "C"
// CHECK:           hextir.for "parallel" %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             hextir.for "thread_binding" %{{.*}} to %{{.*}} step %{{.*}} bind "threadIdx.x" {
// CHECK:               %{{.*}} = hextir.buffer_load %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x2xf64> -> f64
// CHECK:               hextir.buffer_store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : f64, memref<2x2xf64>
// CHECK:         hextir.return
