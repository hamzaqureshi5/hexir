func.func @licm(%x: f64, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %p = ninja.pure_op %x
    %m = ninja.nomem %x
    %s = ninja.spec %x
    "test.use"(%p, %m, %s) : (f64, f64, f64) -> ()
  }
  return
}
