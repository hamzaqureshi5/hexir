func.func @fold() -> f64 {
  %a = ninja.constant 1.0
  %b = ninja.constant 2.0
  %c = ninja.add %a, %b : f64
  return %c : f64
}
func.func @strike_once(%x: f64) -> f64 {
  %y = ninja.strike %x : f64 times 1 : f64
  return %y : f64
}
func.func @strike_twice(%x: f64) -> f64 {
  %y = ninja.strike %x : f64 times 2 : f64
  return %y : f64
}
func.func @commute(%x: f64) -> f64 {
  %k = ninja.constant 5.0
  %y = ninja.add %k, %x : f64
  return %y : f64
}
func.func @unused(%x: f64) {
  %y = ninja.add %x, %x : f64
  return
}
