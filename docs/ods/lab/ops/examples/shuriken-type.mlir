func.func @f() {
  %s = ninja.forge : !ninja.shuriken<4>
  ninja.throw %s : !ninja.shuriken<4>
  return
}
