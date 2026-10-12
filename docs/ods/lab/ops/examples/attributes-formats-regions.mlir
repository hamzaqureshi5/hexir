func.func @main(%a: f64, %b: f64, %c: f64, %i: i32) {
  ninja.dodge "left"
  ninja.dodge "right" steps 3
  ninja.dodge "up" fast
  ninja.dodge "down" steps 2 fast speed 4.0
  ninja.dodge "back" speed 1.0
  ninja.shadow 5
  %s = ninja.sum %a, %b, %c
  %m = ninja.mix %a, %i : (f64, i32) -> i32
  ninja.stance attack
  %star = ninja.forge : !ninja.shuriken<4>
  ninja.throw %star : !ninja.shuriken<4>
  ninja.dojo "hall" {
    %z = ninja.constant 1.0
  }
  ninja.dojo "garden" attributes {open} {
  }
  return
}
