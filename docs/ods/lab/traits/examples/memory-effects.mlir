func.func @cse(%x: f64) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
  %p1 = ninja.pure_op %x
  %p2 = ninja.pure_op %x
  %n1 = ninja.nomem %x
  %n2 = ninja.nomem %x
  %w1 = ninja.writes %x
  %w2 = ninja.writes %x
  %i1 = ninja.noinfo %x
  %i2 = ninja.noinfo %x
  return %p1, %p2, %n1, %n2, %w1, %w2, %i1, %i2 : f64, f64, f64, f64, f64, f64, f64, f64
}
func.func @dce(%x: f64) {
  %p = ninja.pure_op %x
  %n = ninja.nomem %x
  %w = ninja.writes %x
  %i = ninja.noinfo %x
  return
}
func.func @reads(%m: memref<4xf64>, %v: f64) -> (f64, f64, f64) {
  %a = ninja.reads %m : memref<4xf64>
  %b = ninja.reads %m : memref<4xf64>
  ninja.stores %v, %m : memref<4xf64>
  %c = ninja.reads %m : memref<4xf64>
  return %a, %b, %c : f64, f64, f64
}
func.func @boxes(%x: f64) {
  ninja.box { %p = ninja.pure_op %x }
  ninja.box { %w = ninja.writes %x }
  ninja.plainbox { %p = ninja.pure_op %x }
  return
}
