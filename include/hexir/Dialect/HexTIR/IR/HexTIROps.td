//===- HexTIROps.td - Hexir kernel-level IR -----------------*- tablegen -*-===//
//
// `hextir` is the kernel level of the Hexir stack. Where the `hexir` dialect
// answers "what to compute, and where", `hextir` answers "how one device
// computes it": prim funcs over buffers, schedulable blocks, and explicit loop
// nests whose kind (serial / parallel / vectorized / thread_binding) is the
// scheduling decision.
//
// It is deliberately thin. It is not a reimplementation of a scheduling
// language -- `hextir.for` and `hextir.block` carry the decisions, and lowering
// hands the loop nests to `scf` / `memref` / `gpu`, which already have the
// tiling and vectorization machinery.
//
//===----------------------------------------------------------------------===//

#ifndef HEXTIR_OPS
#define HEXTIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def HexTIR_Dialect : Dialect {
  let name = "hextir";
  let cppNamespace = "::mlir::hextir";
  let summary = "kernel-level IR: prim funcs, blocks, loops and buffers";
  let description = [{
    The `hextir` dialect represents a single device kernel after placement has
    been decided. Values are buffers (memrefs), not tensors: a prim func is
    destination-passing and returns nothing, so its caller
    (`hexir.call_tir`) is the level that still speaks in tensors.
  }];
}

class HexTIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<HexTIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// PrimFuncOp
//===----------------------------------------------------------------------===//

def HexTIR_PrimFuncOp : HexTIR_Op<"prim_func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "a single device kernel over buffers";
  let description = [{
    The unit of code generation. Arguments are buffers; the function is
    destination-passing, so results are written into an output buffer argument
    rather than returned.

    ```mlir
    hextir.prim_func @matmul(%a: memref<2x2xf64>,
                             %b: memref<2x2xf64>,
                             %c: memref<2x2xf64>) {
      hextir.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def HexTIR_ReturnOp : HexTIR_Op<"return", [
    Pure, Terminator, ReturnLike, HasParent<"::mlir::hextir::PrimFuncOp">
  ]> {
  let summary = "terminator of a prim func";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
}

//===----------------------------------------------------------------------===//
// Schedulable block
//===----------------------------------------------------------------------===//

def HexTIR_BlockOp : HexTIR_Op<"block", [
    SingleBlockImplicitTerminator<"::mlir::hextir::YieldOp">,
    RecursiveMemoryEffects
  ]> {
  let summary = "a named, schedulable region of computation";
  let description = [{
    The unit a scheduler addresses by name. Wrapping the loop nest of one
    logical computation in a named block is what lets a scheduling pass say
    "tile block @C by 32" without pattern-matching on loop structure.

    ```mlir
    hextir.block "C" {
      hextir.for "serial" %i = %c0 to %cM step %c1 { ... }
      hextir.yield
    }
    ```
  }];
  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$name attr-dict-with-keyword $body";
}

//===----------------------------------------------------------------------===//
// Loops
//===----------------------------------------------------------------------===//

def HexTIR_ForOp : HexTIR_Op<"for", [
    SingleBlockImplicitTerminator<"::mlir::hextir::YieldOp">,
    RecursiveMemoryEffects
  ]> {
  let summary = "a loop nest level carrying its scheduling decision";
  let description = [{
    `kind` is the scheduling decision for this loop level, and is the single
    place a schedule is recorded:

      * `serial`         -- sequential
      * `parallel`       -- may run concurrently (CPU threads)
      * `vectorized`     -- lower to vector ops
      * `unrolled`       -- fully unroll
      * `thread_binding` -- bind to a GPU thread/block axis named by `thread`

    ```mlir
    hextir.for "thread_binding" %c0 to %cN step %c1 bind "threadIdx.x" {
      hextir.yield
    }
    ```
  }];
  let arguments = (ins
    Index:$lower_bound,
    Index:$upper_bound,
    Index:$step,
    StrAttr:$kind,
    OptionalAttr<StrAttr>:$thread,
    Variadic<AnyType>:$init_args
  );
  // Loop-carried values. A reduction accumulator belongs in a register, not in
  // the buffer it eventually lands in: without this the only place to keep a
  // running sum is memory, and the inner loop pays a load and a store per
  // iteration for something that never needs to leave the thread.
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let builders = [OpBuilder<(ins
    "Value":$lowerBound, "Value":$upperBound, "Value":$step, "StringRef":$kind,
    CArg<"ValueRange", "{}">:$initArgs)
  >];

  let extraClassDeclaration = [{
    /// The loop induction variable (the first entry block argument).
    ::mlir::Value getInductionVar() { return getBody().front().getArgument(0); }

    /// The loop-carried values, one per init arg, after the induction variable.
    ::mlir::Block::BlockArgListType getRegionIterArgs() {
      return getBody().front().getArguments().drop_front();
    }
  }];

  let assemblyFormat = [{
    $kind $lower_bound `to` $upper_bound `step` $step
    (`bind` $thread^)?
    (`iter_args` `(` $init_args^ `:` type($init_args) `)`)?
    attr-dict-with-keyword $body (`->` type($results)^)?
  }];
}

def HexTIR_YieldOp : HexTIR_Op<"yield", [Pure, Terminator]> {
  let summary = "terminator of a hextir.for or hextir.block region";
  let arguments = (ins Variadic<AnyType>:$values);
  // Zero-arg builder so SingleBlockImplicitTerminator can materialize it.
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
}

//===----------------------------------------------------------------------===//
// Buffers
//===----------------------------------------------------------------------===//

def HexTIR_AllocBufferOp : HexTIR_Op<"alloc_buffer"> {
  let summary = "allocate a kernel-local buffer";
  let description = [{
    A buffer local to the kernel. The memory space in the memref type is what
    distinguishes scratch from shared/workgroup memory, so promoting a buffer
    to GPU shared memory is a type change on this op, not a new op.
  }];
  let results = (outs AnyMemRef:$buffer);
  let assemblyFormat = "attr-dict `:` type($buffer)";
}

def HexTIR_BufferLoadOp : HexTIR_Op<"buffer_load"> {
  let summary = "read one element of a buffer";
  let arguments = (ins AnyMemRef:$buffer, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $buffer `[` $indices `]` attr-dict `:` type($buffer) `->` type($result)
  }];
}

def HexTIR_BufferStoreOp : HexTIR_Op<"buffer_store"> {
  let summary = "write one element of a buffer";
  let arguments = (ins AnyType:$value, AnyMemRef:$buffer, Variadic<Index>:$indices);
  let assemblyFormat = [{
    $value `,` $buffer `[` $indices `]` attr-dict
    `:` type($value) `,` type($buffer)
  }];
}

#endif // HEXTIR_OPS
