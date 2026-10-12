// Part A: examples, one op per attribute kind.

def Ninja_KickOp : Ninja_Op<"kick"> {
  let arguments = (ins I32Attr:$power);
  let assemblyFormat = "$power attr-dict";
}
def Ninja_RunOp : Ninja_Op<"run"> {
  let arguments = (ins I64Attr:$meters);
  let assemblyFormat = "$meters attr-dict";
}
def Ninja_StepOp : Ninja_Op<"step"> {
  let arguments = (ins SI32Attr:$offset);
  let assemblyFormat = "$offset attr-dict";
}
def Ninja_CountOp : Ninja_Op<"count"> {
  let arguments = (ins UI32Attr:$amount);
  let assemblyFormat = "$amount attr-dict";
}
def Ninja_AimOp : Ninja_Op<"aim"> {
  let arguments = (ins F32Attr:$accuracy);
  let assemblyFormat = "$accuracy attr-dict";
}
def Ninja_HideOp : Ninja_Op<"hide"> {
  let arguments = (ins BoolAttr:$visible);
  let assemblyFormat = "$visible attr-dict";
}
def Ninja_SayOp : Ninja_Op<"say"> {
  let arguments = (ins StrAttr:$words);
  let assemblyFormat = "$words attr-dict";
}
def Ninja_SneakOp : Ninja_Op<"sneak"> {
  let arguments = (ins UnitAttr:$quiet);
  let assemblyFormat = "(`quiet` $quiet^)? attr-dict";
}
def Ninja_ForgeAsOp : Ninja_Op<"forge_as"> {
  let arguments = (ins TypeAttr:$kind);
  let assemblyFormat = "$kind attr-dict";
}
def Ninja_BagOp : Ninja_Op<"bag"> {
  let arguments = (ins ArrayAttr:$items);
  let assemblyFormat = "$items attr-dict";
}
def Ninja_TeamOp : Ninja_Op<"team"> {
  let arguments = (ins StrArrayAttr:$members);
  let assemblyFormat = "$members attr-dict";
}
def Ninja_PathOp : Ninja_Op<"path"> {
  let arguments = (ins DenseI64ArrayAttr:$steps);
  let assemblyFormat = "$steps attr-dict";
}
def Ninja_ProfileOp : Ninja_Op<"profile"> {
  let arguments = (ins DictionaryAttr:$info);
  let assemblyFormat = "$info attr-dict";
}
def Ninja_MapOp : Ninja_Op<"map"> {
  let arguments = (ins ElementsAttr:$tiles);
  let assemblyFormat = "$tiles attr-dict";
}
def Ninja_TechniqueOp : Ninja_Op<"technique"> {
  let arguments = (ins SymbolNameAttr:$name);
  let assemblyFormat = "$name attr-dict";
}
def Ninja_UseOp : Ninja_Op<"use"> {
  let arguments = (ins FlatSymbolRefAttr:$technique);
  let assemblyFormat = "$technique attr-dict";
}
def Ninja_NoteOp : Ninja_Op<"note"> {
  let arguments = (ins AnyAttr:$text);
  let assemblyFormat = "$text attr-dict";
}
def Ninja_ValueOp : Ninja_Op<"value"> {
  let arguments = (ins TypedAttrInterface:$v);
  let assemblyFormat = "$v attr-dict";
}
def Ninja_ClimbOp : Ninja_Op<"climb"> {
  let arguments = (ins I32Attr:$height, OptionalAttr<I32Attr>:$rope);
  let assemblyFormat = "$height (`rope` $rope^)? attr-dict";
}
def Ninja_ThrowOp : Ninja_Op<"throw"> {
  let arguments = (ins DefaultValuedAttr<I32Attr, "1">:$count);
  let assemblyFormat = "(`count` $count^)? attr-dict";
}
def Ninja_HealOp : Ninja_Op<"heal"> {
  let arguments = (ins ConfinedAttr<I32Attr, [IntMinValue<1>, IntMaxValue<100>]>:$points);
  let assemblyFormat = "$points attr-dict";
}
