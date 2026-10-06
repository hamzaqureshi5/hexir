// The ninja verifiers reject bad IR, with a source location.
//
// RUN: not %ninja-opt %s 2>&1 | FileCheck %s

func.func @main() -> f64 {
  %x = ninja.constant 2.5
  // CHECK: ninja-invalid.mlir:[[@LINE+1]]:8: error: 'ninja.strike' op times must be positive, got 0
  %z = ninja.strike %x : f64 times 0 : f64
  return %z : f64
}
