// The ninja dialect (a toy for learning ODS) parses, verifies and prints.
//
// RUN: %ninja-opt %s 2>&1 | FileCheck %s

module {
  func.func @main() -> f64 {
    // CHECK: %[[X:.*]] = ninja.constant 2.500000e+00
    %x = ninja.constant 2.5
    // CHECK: %[[Y:.*]] = ninja.add %[[X]], %[[X]] : f64
    %y = ninja.add %x, %x : f64
    // CHECK: %[[Z:.*]] = ninja.strike %[[Y]] : f64 times 3 : f64
    %z = ninja.strike %y : f64 times 3 : f64
    return %z : f64
  }
}
