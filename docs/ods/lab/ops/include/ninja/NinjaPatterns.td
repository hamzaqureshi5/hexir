include "mlir/IR/PatternBase.td"
include "ninja/NinjaOps.td"

// x + x  ==>  strike x times 2
def AddSelfToStrike : Pat<(Ninja_AddOp F64:$x, $x),
                          (Ninja_StrikeOp $x, ConstantAttr<I32Attr, "2">)>;
