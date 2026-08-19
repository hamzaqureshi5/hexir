// A square matmul with no print, so the benchmark measures compute rather than
// formatting a million numbers to stdout. Splat constants keep the file small.
func.func @main() {
  %a = hexir.constant dense<1.000000e+00> : tensor<512x512xf64>
  %b = hexir.constant dense<2.000000e+00> : tensor<512x512xf64>
  %m = hexir.linear %a, %b : tensor<512x512xf64>
  return
}
