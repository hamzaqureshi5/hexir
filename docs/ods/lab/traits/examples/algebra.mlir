func.func @commutative(%x: f64) -> f64 {
  %k = arith.constant 5.0 : f64
  %y = ninja.mul %k, %x : f64
  return %y : f64
}
func.func @idempotent_unary(%x: f64) -> f64 {
  %a = ninja.abs %x : f64
  %b = ninja.abs %a : f64
  return %b : f64
}
func.func @idempotent_binary(%x: f64) -> f64 {
  %m = ninja.max %x, %x : f64
  return %m : f64
}
func.func @involution(%x: f64) -> f64 {
  %a = ninja.neg %x : f64
  %b = ninja.neg %a : f64
  return %b : f64
}
