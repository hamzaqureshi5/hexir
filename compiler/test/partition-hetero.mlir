// PartitionPass + MaterializeLSTargetsPass: the module is tagged with its
// targets, matmul is placed on the GPU (ls_gpu) and relu on the CPU (ls_cpu).
// RUN: %hexir -emit=mlir-hetero 2>&1 | FileCheck %s

// CHECK:       module attributes {hexir.targets = ["cpu", "cuda"]}
// CHECK-LABEL: func.func @main()
// CHECK:         %[[MATMUL:.*]] = ls_gpu.matmul %{{.*}}, %{{.*}} : tensor<2x2xf64>
// CHECK:         %[[RELU:.*]] = ls_cpu.relu %[[MATMUL]] : tensor<2x2xf64>
// CHECK:         hexir.print %{{.*}} {device = "cpu"}
