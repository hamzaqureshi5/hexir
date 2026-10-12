// Lab ops: one op per trait, to verify the guide's examples.

//===--- A. Algebra -------------------------------------------------------===//
def L_MulOp : Ninja_Op<"mul", [Pure, Commutative, SameOperandsAndResultType]> {
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs);
  let results = (outs AnyFloat:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
def L_AbsOp : Ninja_Op<"abs", [Pure, Idempotent, SameOperandsAndResultType]> {
  let arguments = (ins AnyFloat:$input);
  let results = (outs AnyFloat:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}
def L_MaxOp : Ninja_Op<"max", [Pure, Idempotent, SameOperandsAndResultType]> {
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs);
  let results = (outs AnyFloat:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
def L_NegOp : Ninja_Op<"neg", [Pure, Involution, SameOperandsAndResultType]> {
  let arguments = (ins AnyFloat:$input);
  let results = (outs AnyFloat:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

//===--- B. Built-in type relations ---------------------------------------===//
def L_PairOp : Ninja_Op<"pair", [SameTypeOperands]> {
  let arguments = (ins AnyType:$a, AnyType:$b);
  let results = (outs AnyType:$result);
}
def L_ZipOp : Ninja_Op<"zip", [SameOperandsShape]> {
  let arguments = (ins AnyTensor:$a, AnyTensor:$b);
  let results = (outs AnyTensor:$result);
}
def L_ConvertOp : Ninja_Op<"convert", [SameOperandsAndResultShape]> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}
def L_ConcatOp : Ninja_Op<"concat", [SameOperandsElementType]> {
  let arguments = (ins AnyTensor:$a, AnyTensor:$b);
  let results = (outs AnyTensor:$result);
}
def L_ReshapeOp : Ninja_Op<"reshape", [SameOperandsAndResultElementType]> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}
def L_TransposeOp : Ninja_Op<"transpose", [SameOperandsAndResultRank]> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}
def L_ToFloatOp : Ninja_Op<"to_float", [ResultsAreFloatLike]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def L_BAddOp : Ninja_Op<"badd", [ResultsBroadcastableShape]> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
}
def L_TypedConstOp : Ninja_Op<"typed_const", [FirstAttrDerivedResultType]> {
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
}

//===--- C. Predicate traits (name the values to compare) -----------------===//
def L_SelectOp : Ninja_Op<"select", [AllTypesMatch<["a", "b", "result"]>]> {
  let arguments = (ins I1:$cond, AnyType:$a, AnyType:$b);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$cond `,` $a `,` $b attr-dict `:` type($result)";
}
def L_StackOp : Ninja_Op<"stack", [AllElementTypesMatch<["a", "b"]>]> {
  let arguments = (ins AnyTensor:$a, AnyTensor:$b);
  let results = (outs AnyTensor:$result);
}
def L_BitcastOp : Ninja_Op<"bitcast", [AllShapesMatch<["input", "result"]>]> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}
def L_PermuteOp : Ninja_Op<"permute", [AllRanksMatch<["input", "result"]>]> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}
def L_FlattenOp : Ninja_Op<"flatten", [AllElementCountsMatch<["input", "result"]>]> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}
def L_SameKindOp : Ninja_Op<"same_kind", [AllTypeIDsMatch<["a", "b"]>]> {
  let arguments = (ins AnyType:$a, AnyType:$b);
}
def L_ExtractOp : Ninja_Op<"extract", [
    TypesMatchWith<"result is the element type of input", "input", "result",
                   "::llvm::cast<::mlir::ShapedType>($_self).getElementType()">]> {
  let arguments = (ins AnyRankedTensor:$input, Index:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `[` $index `]` attr-dict `:` type($input)";
}
def L_ExtractOrOp : Ninja_Op<"extract_or", [
    OptionalTypesMatchWith<"fallback is the element type of input", "input", "fallback",
                   "::llvm::cast<::mlir::ShapedType>($_self).getElementType()">]> {
  let arguments = (ins AnyRankedTensor:$input, Optional<AnyType>:$fallback);
  let results = (outs AnyType:$result);
}
def L_CopyAllOp : Ninja_Op<"copy_all", [
    RangedTypesMatchWith<"result types match input types", "inputs", "results", "$_self">]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
}
def L_CountOp : Ninja_Op<"count", [TypeIs<"result", I32>]> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyType:$result);
}
def L_ReluOp : Ninja_Op<"relu", [ElementTypeIs<"input", F32>]> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}
def L_FromElementsOp : Ninja_Op<"from_elements", [
    ShapedTypeMatchesElementCountAndTypes<"result", "elements">]> {
  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs AnyRankedTensor:$result);
}
def L_ScaleByOp : Ninja_Op<"scale_by", [InputMatchesTypes<["value", "factor"], [F32, I32]>]> {
  let arguments = (ins AnyType:$value, AnyType:$factor);
}
def L_ComboOp : Ninja_Op<"combo", [
    InputAddressIsCombinationOf<["a", "b"], [[F32, F32], [F64, I64]]>]> {
  let arguments = (ins AnyType:$a, AnyType:$b);
}
def L_MatmulOp : Ninja_Op<"matmul", [
    PredOpTrait<"lhs is a rank-2 tensor",
                CPred<"::llvm::cast<::mlir::ShapedType>($lhs.getType()).getRank() == 2">>]> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
}
def L_AllMatchOp : Ninja_Op<"all_match", [
    AllMatch<["$a.getType()", "$b.getType()"], "a and b have the same type">]> {
  let arguments = (ins AnyType:$a, AnyType:$b);
}
def L_SameWidthOp : Ninja_Op<"same_width", [
    AllMatchSameOperatorTrait<["a", "b"], "$_self.getType().getIntOrFloatBitWidth()",
                              "bit width">]> {
  let arguments = (ins AnySignlessInteger:$a, AnySignlessInteger:$b);
}
def L_AnyFloatOp : Ninja_Op<"any_float", [
    AnyMatchOperatorTrait<["a", "b"], "::llvm::isa<::mlir::FloatType>($_self.getType())",
                          "float type">]> {
  let arguments = (ins AnyType:$a, AnyType:$b);
}

//===--- D. Elementwise family --------------------------------------------===//
def L_EltOp : Ninja_Op<"elt", [Elementwise]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def L_SigmoidOp : Ninja_Op<"sigmoid", [Pure, ElementwiseMappable]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def L_ScalOp : Ninja_Op<"scal", [Elementwise, Scalarizable]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def L_VecOp : Ninja_Op<"vec", [Elementwise, Vectorizable]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def L_TensOp : Ninja_Op<"tens", [Elementwise, Tensorizable]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}

//===--- E. Structure -----------------------------------------------------===//
def L_DojoOp : Ninja_Op<"dojo", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}
def L_YieldOp : Ninja_Op<"yield", [Pure, Terminator,
                                    ParentOneOf<["DojoOp", "TempleOp"]>]> {
  let assemblyFormat = "attr-dict";
}
def L_TempleOp : Ninja_Op<"temple", []> {
  let regions = (region AnyRegion:$body);
}
def L_ScrollOp : Ninja_Op<"scroll", [NoTerminator, SingleBlock]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}
def L_BlockOnlyOp : Ninja_Op<"one_block", [SingleBlock]> {
  let regions = (region AnyRegion:$body);
}
def L_NoArgsOp : Ninja_Op<"no_args", [NoRegionArguments, NoTerminator, SingleBlock]> {
  let regions = (region AnyRegion:$body);
}
def L_SealedOp : Ninja_Op<"sealed", [IsolatedFromAbove, NoTerminator, SingleBlock]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}
def L_BowOp : Ninja_Op<"bow", [HasParent<"DojoOp">]> {
  let assemblyFormat = "attr-dict";
}
def L_MeditateOp : Ninja_Op<"meditate", [HasAncestor<"DojoOp">]> {
  let assemblyFormat = "attr-dict";
}
def L_RestOp : Ninja_Op<"rest", [AncestorOneOf<["DojoOp", "TempleOp"]>]> {
  let assemblyFormat = "attr-dict";
}

//===--- F. Variadic sizing -----------------------------------------------===//
def L_ZipListsOp : Ninja_Op<"zip_lists", [SameVariadicOperandSize]> {
  let arguments = (ins Variadic<F32>:$xs, Variadic<F32>:$ys);
  let assemblyFormat = "`(` $xs `)` `(` $ys `)` attr-dict";
  let hasVerifier = 1;
}
def L_SplitOp : Ninja_Op<"split", [SameVariadicResultSize]> {
  let arguments = (ins I32:$n);
  let results = (outs Variadic<F32>:$evens, Variadic<F32>:$odds);
}
def L_ThrowAtOp : Ninja_Op<"throw_at", [AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<F32>:$targets, Optional<I32>:$times);
  let assemblyFormat = "`(` $targets `)` (`times` $times^)? attr-dict";
}
def L_UnpackOp : Ninja_Op<"unpack", [AttrSizedResultSegments]> {
  let arguments = (ins I32:$n);
  let results = (outs Variadic<F32>:$floats, Optional<I32>:$extra);
}

//===--- G. Scopes and markers --------------------------------------------===//
def L_ScopeOp : Ninja_Op<"scope", [AutomaticAllocationScope, NoTerminator, SingleBlock]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}
def L_MakeTokenOp : Ninja_Op<"make_token", [TokenProducerTrait]> {
  let results = (outs Token:$token);
  let assemblyFormat = "attr-dict";
}
def L_UseTokenOp : Ninja_Op<"use_token", [TokenConsumerTrait]> {
  let arguments = (ins Token:$token);
  let assemblyFormat = "$token attr-dict";
}
def L_PlainUseOp : Ninja_Op<"plain_use", []> {
  let arguments = (ins Token:$token);
  let assemblyFormat = "$token attr-dict";
}

//===--- H. Symbols and region kinds --------------------------------------===//
def L_ClanOp : Ninja_Op<"clan", [SymbolTable, NoTerminator, SingleBlock]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}
def L_MemberOp : Ninja_Op<"member", [Symbol, SymbolName, SymbolVisibility]> {
  let arguments = (ins SymbolNameAttr:$sym_name, OptionalAttr<StrAttr>:$sym_visibility);
  let assemblyFormat = "($sym_visibility^)? $sym_name attr-dict";
}
def L_GraphOp : Ninja_Op<"graph", [
    DeclareOpInterfaceMethods<RegionKindInterface>, NoTerminator, SingleBlock]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}
def L_GraphOnlyOp : Ninja_Op<"graph_only", [HasOnlyGraphRegion, NoTerminator, SingleBlock]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}
def L_GraphNoTermOp : Ninja_Op<"graph_noterm", [GraphRegionNoTerminator]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}
def L_FakeTokenOp : Ninja_Op<"fake_token", []> {
  let results = (outs Token:$token);
  let assemblyFormat = "attr-dict";
}
def L_AnythingOp : Ninja_Op<"anything", []> {
  let results = (outs Type<CPred<"true">>:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}
def L_AnythingProducerOp : Ninja_Op<"anything_producer", [TokenProducerTrait]> {
  let results = (outs Type<CPred<"true">>:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}
def L_GraphOnly2Op : Ninja_Op<"graph_only2", [RegionKindInterface, HasOnlyGraphRegion,
                                               NoTerminator, SingleBlock]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

//===--- I. Interface traits ----------------------------------------------===//
def L_PureOp : Ninja_Op<"pure_op", [Pure]> {
  let arguments = (ins F64:$x);
  let results = (outs F64:$r);
  let assemblyFormat = "$x attr-dict";
}
def L_NoMemOp : Ninja_Op<"nomem", [NoMemoryEffect]> {
  let arguments = (ins F64:$x);
  let results = (outs F64:$r);
  let assemblyFormat = "$x attr-dict";
}
def L_WritesOp : Ninja_Op<"writes", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins F64:$x);
  let results = (outs F64:$r);
  let assemblyFormat = "$x attr-dict";
}
def L_NoInfoOp : Ninja_Op<"noinfo", []> {
  let arguments = (ins F64:$x);
  let results = (outs F64:$r);
  let assemblyFormat = "$x attr-dict";
}
def L_ReadsOp : Ninja_Op<"reads", []> {
  let arguments = (ins Arg<AnyMemRef, "the buffer", [MemRead]>:$buf);
  let results = (outs F64:$r);
  let assemblyFormat = "$buf attr-dict `:` type($buf)";
}
def L_StoresOp : Ninja_Op<"stores", []> {
  let arguments = (ins F64:$value, Arg<AnyMemRef, "the buffer", [MemWrite]>:$buf);
  let assemblyFormat = "$value `,` $buf attr-dict `:` type($buf)";
}
def L_BoxOp : Ninja_Op<"box", [RecursiveMemoryEffects, NoTerminator, SingleBlock]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}
def L_PlainBoxOp : Ninja_Op<"plainbox", [NoTerminator, SingleBlock]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}
def L_ElemOp : Ninja_Op<"elem", [Pure, InferTypeOpAdaptor]> {
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
def L_NamedOp : Ninja_Op<"named", [Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let arguments = (ins F64:$lhs, F64:$rhs);
  let results = (outs F64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}
def L_CastOp : Ninja_Op<"cast", [Pure, DeclareOpInterfaceMethods<CastOpInterface>]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
}
def L_SpecOp : Ninja_Op<"spec", [NoMemoryEffect, AlwaysSpeculatable]> {
  let arguments = (ins F64:$x);
  let results = (outs F64:$r);
  let assemblyFormat = "$x attr-dict";
}
def L_RankOp : Ninja_Op<"rank", []> {
  let arguments = (ins ConfinedAttr<I64Attr, [IntMinValue<1>, IntMaxValue<10>]>:$level,
                       DenseI64ArrayAttr:$scores,
                       ConfinedAttr<StrArrayAttr, [ArrayMinCount<1>]>:$titles,
                       SI32Attr:$offset);
  let assemblyFormat = "$level `scores` $scores `titles` $titles `offset` $offset attr-dict";
}
def L_ArenaOp : Ninja_Op<"arena", [NoTerminator, SingleBlock]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasRegionVerifier = 1;
}
def L_ChoiceOp : Ninja_Op<"choice", [NoTerminator]> {
  let arguments = (ins I1:$cond);
  let regions = (region SizedRegion<1>:$then, MaxSizedRegion<1>:$otherwise);
  let assemblyFormat = "$cond $then `else` $otherwise attr-dict";
}
def L_ManyOp : Ninja_Op<"many", [NoTerminator, SingleBlock]> {
  let regions = (region VariadicRegion<AnyRegion>:$parts);
}
def L_JumpOp : Ninja_Op<"jump_to", [Terminator, DeclareOpInterfaceMethods<BranchOpInterface>]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = "$dest (`(` $args^ `:` type($args) `)`)? attr-dict";
}
def L_Mix2Op : Ninja_Op<"mix2"> {
  let arguments = (ins AnyType:$a, AnyType:$b);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$a `,` $b attr-dict `:` functional-type(operands, results)";
}
