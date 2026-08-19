// LowerToLinalgPass: hexir ops become linalg/arith on tensors.
// hexir.linear -> linalg.matmul, hexir.relu -> linalg.generic with maximumf.
//
// hexir.print survives to the LLVM stage, so it is still here.
//
// RUN: %hexir -emit=mlir-linalg %s 2>&1 | FileCheck %s

func.func @main() {
  %a = hexir.constant dense<[[3.0, 1.0], [2.0, 2.0]]> : tensor<2x2xf64>
  %b = hexir.constant dense<[[1.0, 5.0], [5.0, 2.0]]> : tensor<2x2xf64>
  %m = hexir.linear %a, %b : tensor<2x2xf64>
  %r = hexir.relu %m : tensor<2x2xf64>
  hexir.print %r : tensor<2x2xf64>
  return
}

// CHECK-LABEL: func.func @main()
// CHECK-NOT:     hexir.linear

// The matmul destination is tensor.empty + linalg.fill, never a constant: a
// constant is read-only, so bufferization would have to copy it to make the
// destination writable.
// CHECK:         %[[EMPTY:.*]] = tensor.empty()
// CHECK:         %[[INIT:.*]] = linalg.fill ins(%{{.*}} : f64) outs(%[[EMPTY]] : tensor<2x2xf64>)
// CHECK:         %[[MATMUL:.*]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<2x2xf64>, tensor<2x2xf64>) outs(%[[INIT]]

// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[MATMUL]] : tensor<2x2xf64>)
// CHECK:           arith.maximumf
// CHECK:           linalg.yield
// CHECK-NOT:     hexir.relu
// CHECK:         hexir.print
