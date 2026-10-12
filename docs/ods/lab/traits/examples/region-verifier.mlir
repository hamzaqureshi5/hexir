func.func @f(%x: f64) {
  ninja.arena {
    %y = ninja.neg %x : f64
  }
  ninja.arena {
  }
  return
}
