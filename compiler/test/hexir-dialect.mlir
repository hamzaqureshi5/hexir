// Initial IR: the synthetic program from Builder.cpp in the hexir dialect.
// RUN: %hexir -emit=mlir 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @main()
// CHECK:         %{{.*}} = hexir.constant dense<{{.*}}> : tensor<2x2xf64>
// CHECK:         %[[LINEAR:.*]] = hexir.linear %{{.*}}, %{{.*}} : tensor<2x2xf64>
// CHECK:         %[[RELU:.*]] = hexir.relu %[[LINEAR]] : tensor<2x2xf64>
// CHECK:         hexir.print %[[RELU]] : tensor<2x2xf64>
// CHECK:         return
