// LowerToLinalgPass: hexir ops become linalg/arith on tensors.
// hexir.linear -> linalg.matmul, hexir.relu -> linalg.generic with maximumf.
// RUN: %hexir -emit=mlir-linalg 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @main()
// CHECK-NOT:     hexir.linear
// CHECK:         %[[MATMUL:.*]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<2x2xf64>, tensor<2x2xf64>)
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[MATMUL]] : tensor<2x2xf64>)
// CHECK:           arith.maximumf
// CHECK:           linalg.yield
// CHECK-NOT:     hexir.relu
// CHECK:         hexir.print
