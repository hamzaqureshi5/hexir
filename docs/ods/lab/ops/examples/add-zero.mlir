func.func @add_zero(%x: f64) -> f64 {
  %z = ninja.constant 0.0
  %y = ninja.add %x, %z : f64
  return %y : f64
}
func.func @zero_first(%x: f64) -> f64 {
  %z = ninja.constant 0.0
  %y = ninja.add %z, %x : f64
  return %y : f64
}
