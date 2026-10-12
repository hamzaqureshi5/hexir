func.func @twice(%x: f64) -> f64 {
  %y = ninja.add %x, %x : f64
  return %y : f64
}
func.func @f32(%x: f32) -> f32 {
  %y = ninja.add %x, %x : f32
  return %y : f32
}
