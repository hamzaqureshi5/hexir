// Part B: drill solutions.
def Ninja_PunchOp : Ninja_Op<"punch"> {
  let arguments = (ins I32Attr:$force);
  let assemblyFormat = "$force attr-dict";
}
def Ninja_WaitOp : Ninja_Op<"wait"> {
  let arguments = (ins I64Attr:$millis);
  let assemblyFormat = "$millis attr-dict";
}
def Ninja_TurnOp : Ninja_Op<"turn"> {
  let arguments = (ins SI32Attr:$degrees);
  let assemblyFormat = "$degrees attr-dict";
}
def Ninja_ArrowsOp : Ninja_Op<"arrows"> {
  let arguments = (ins UI32Attr:$amount);
  let assemblyFormat = "$amount attr-dict";
}
def Ninja_BalanceOp : Ninja_Op<"balance"> {
  let arguments = (ins F32Attr:$level);
  let assemblyFormat = "$level attr-dict";
}
def Ninja_DisguiseOp : Ninja_Op<"disguise"> {
  let arguments = (ins BoolAttr:$on);
  let assemblyFormat = "$on attr-dict";
}
def Ninja_WhisperOp : Ninja_Op<"whisper"> {
  let arguments = (ins StrAttr:$message);
  let assemblyFormat = "$message attr-dict";
}
def Ninja_VanishOp : Ninja_Op<"vanish"> {
  let arguments = (ins UnitAttr:$silent);
  let assemblyFormat = "(`silent` $silent^)? attr-dict";
}
def Ninja_ConvertToOp : Ninja_Op<"convert_to"> {
  let arguments = (ins TypeAttr:$target);
  let assemblyFormat = "$target attr-dict";
}
def Ninja_PackOp : Ninja_Op<"pack"> {
  let arguments = (ins ArrayAttr:$items);
  let assemblyFormat = "$items attr-dict";
}
def Ninja_AlliesOp : Ninja_Op<"allies"> {
  let arguments = (ins StrArrayAttr:$names);
  let assemblyFormat = "$names attr-dict";
}
def Ninja_RouteOp : Ninja_Op<"route"> {
  let arguments = (ins DenseI64ArrayAttr:$stops);
  let assemblyFormat = "$stops attr-dict";
}
def Ninja_MissionOp : Ninja_Op<"mission"> {
  let arguments = (ins DictionaryAttr:$details);
  let assemblyFormat = "$details attr-dict";
}
def Ninja_GridOp : Ninja_Op<"grid"> {
  let arguments = (ins ElementsAttr:$cells);
  let assemblyFormat = "$cells attr-dict";
}
def Ninja_JutsuOp : Ninja_Op<"jutsu"> {
  let arguments = (ins SymbolNameAttr:$name);
  let assemblyFormat = "$name attr-dict";
}
def Ninja_PerformOp : Ninja_Op<"perform"> {
  let arguments = (ins FlatSymbolRefAttr:$jutsu);
  let assemblyFormat = "$jutsu attr-dict";
}
def Ninja_TagOp : Ninja_Op<"tag"> {
  let arguments = (ins AnyAttr:$label);
  let assemblyFormat = "$label attr-dict";
}
def Ninja_SealOp : Ninja_Op<"seal"> {
  let arguments = (ins TypedAttrInterface:$code);
  let assemblyFormat = "$code attr-dict";
}
def Ninja_LeapOp : Ninja_Op<"leap"> {
  let arguments = (ins I32Attr:$distance, OptionalAttr<I32Attr>:$spins);
  let assemblyFormat = "$distance (`spin` $spins^)? attr-dict";
}
def Ninja_BowOp : Ninja_Op<"bow"> {
  let arguments = (ins DefaultValuedAttr<I32Attr, "1">:$times);
  let assemblyFormat = "(`times` $times^)? attr-dict";
}
def Ninja_FocusOp : Ninja_Op<"focus"> {
  let arguments = (ins ConfinedAttr<I32Attr, [IntMinValue<1>, IntMaxValue<10>]>:$level);
  let assemblyFormat = "$level attr-dict";
}
def Ninja_AmbushOp : Ninja_Op<"ambush"> {
  let arguments = (ins
    StrAttr:$target,
    ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<2>]>:$position,
    StrArrayAttr:$allies,
    UnitAttr:$silent,
    OptionalAttr<I64Attr>:$delay
  );
  let assemblyFormat = [{
    $target `at` $position `with` $allies (`silent` $silent^)? (`wait` $delay^)? attr-dict
  }];
  let hasVerifier = 1;
}
