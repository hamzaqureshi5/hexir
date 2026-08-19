// The hexir dialect parses and prints. -emit=mlir runs no passes, so what
// comes back out is exactly what went in, modulo SSA renumbering.
//
// RUN: %hexir -emit=mlir %s 2>&1 | FileCheck %s

func.func @main() {
  %a = hexir.constant dense<[[3.0, 1.0], [2.0, 2.0]]> : tensor<2x2xf64>
  %b = hexir.constant dense<[[1.0, 5.0], [5.0, 2.0]]> : tensor<2x2xf64>
  %m = hexir.linear %a, %b : tensor<2x2xf64>
  %r = hexir.relu %m : tensor<2x2xf64>
  hexir.print %r : tensor<2x2xf64>
  return
}

// CHECK-LABEL: func.func @main()
// CHECK:         %{{.*}} = hexir.constant dense<{{.*}}> : tensor<2x2xf64>
// CHECK:         %[[LINEAR:.*]] = hexir.linear %{{.*}}, %{{.*}} : tensor<2x2xf64>
// CHECK:         %[[RELU:.*]] = hexir.relu %[[LINEAR]] : tensor<2x2xf64>
// CHECK:         hexir.print %[[RELU]] : tensor<2x2xf64>
// CHECK:         return
