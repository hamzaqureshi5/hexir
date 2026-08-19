# Graph transforms

Whole-tensor rewrites, before any loop exists. In Hexir these belong on the
`hexir` dialect, between shape inference and `hexir-lower-to-tir`.

This is where the largest wins for static models are, because it is where
memory traffic is decided. A kernel that reads its input from device memory
instead of from a register has already lost, and no amount of scheduling
recovers it.

## Fusion

Hexir has none. `LowerToTIRPass` emits exactly one `hextir.prim_func` per
`hexir` op, so `linear → add → relu` writes and re-reads two full intermediate
tensors.

| Transform | Prior art | What it buys |
| --- | --- | --- |
| Elementwise into producer | Relay `FuseOps`; XLA `InstructionFusion`; IREE `ElementwiseOpFusion` | The bias-and-activation epilogue stops touching memory. On an MLP this is most of the traffic. |
| Op-kind fusion algebra | Relay's pattern kinds — `kElemWise`, `kBroadcast`, `kInjective`, `kCommReduce`, `kOutEWiseFusable`, `kOpaque` — fused over a dominator tree | A *rule* for what may fuse with what, instead of a list of special cases. The cheapest way to get fusion that generalizes. |
| Multi-output fusion | XLA `MultiOutputFusion` | One kernel producing two live results, e.g. a normalization emitting both output and saved statistics. |
| Horizontal fusion | XLA `HorizontalLoopFusion`, `HorizontalInputFusion`; IREE `FuseHorizontalContractions` | Batches many small independent kernels into one launch. Directly targets decode, where launch count dominates. |
| Fuse into the dispatch region | IREE `FormDispatchRegions` + `CloneProducersIntoDispatchRegions` | Cheap producers get *cloned* into every consumer rather than materialized. Recompute beats a round trip to DRAM. |
| Fusion with a cost model | XLA `PriorityFusion`; `FusionMerger` | Stops fusion that increases work more than it saves bandwidth. Needed once fusion is aggressive. |
| Fuse the kernels after fusing the graph | Relax `FuseOps` then `FuseTIR` | Two-stage: decide at the graph level, then actually merge the loop nests. This is the shape Hexir would want, given `call_tir`. |

Relax's `FuseOps`/`FuseTIR` split is worth singling out. Hexir already has
`hexir.call_tir` and module-scope prim funcs — the same structure Relax uses —
so graph-level fusion can be expressed as "rewrite the calls", with a separate
pass that merges the callee loop nests.

## Algebraic and simplifying rewrites

`HexirCombine.td` exists and every pattern in it is commented out. Only
`ConstantOp::fold` is live.

| Transform | Prior art | What it buys |
| --- | --- | --- |
| Algebraic simplification | XLA `AlgebraicSimplifier`; Relay `SimplifyExpr`, `CanonicalizeOps` | The long tail: add-of-zero, multiply-by-one, `relu(relu(x))`, `transpose(transpose(x))`, reshape chains. |
| Constant folding | Relay `FoldConstant`; XLA `HloConstantFolding` | Already present for single ops. |
| Constant *evaluation* | IREE `HoistIntoGlobals` (const-eval JITs the subgraph at compile time); Relax `LiftTransformParams` | Any subgraph reachable only from weights runs once, at compile time, and the result is baked into RODATA. Folds weight transposes, dequantization setup, and layout changes to nothing. |
| CSE and DCE | Relay `EliminateCommonSubexpr`, `DeadCodeElimination`; XLA `HloCSE`, `HloDCE` | MLIR's `cse` and canonicalizer already cover this at the op level. |
| Batchnorm and inference-only folding | Relay `SimplifyInference`; XLA `BatchNormExpander` | Folds normalization into the preceding matmul's weights. Free at inference. |
| Scale-axis folding | Relay `FoldScaleAxis` (forward and backward) | Pushes a per-channel scale into adjacent weights, removing an elementwise pass. |
| Combining parallel branches | Relay `CombineParallelDense`, `CombineParallelBatchMatmul` | Three projections sharing one input become one wider matmul. This is exactly QKV fusion. |
| Expanding composite ops | XLA `GatherExpander`, `ScatterExpander`, `RngExpander` | Lets the frontend keep high-level ops without every backend needing a kernel for each. |

## Layout and data movement

Hexir has no `transpose` and no `reshape` — both are commented out in
`HexirOps.td`. That is a gap with unusual leverage: layout is often worth more
than scheduling, and for constant weights it is free.

| Transform | Prior art | What it buys |
| --- | --- | --- |
| Global layout assignment | XLA `LayoutAssignment` | Chooses a physical layout per value with constraints, instead of hardcoding row-major everywhere. |
| Layout conversion and alteration | Relay `ConvertLayout`, `AlterOpLayout`; Relax `ConvertLayout` | Inserts the minimum number of transposes to give every kernel the layout it prefers. |
| Transpose propagation | IREE `PropagateLinalgTranspose`; XLA `TransposeFolding`, `ReshapeMover` | Sinks transposes toward the edges of the graph, where they cancel or land on a constant. |
| Reshape as metadata | Relax `RewriteDataflowReshape`; IREE `flow.tensor.reshape` | A reshape becomes a view, never a copy. |
| Weight pre-packing | Relax `LiftTransformParams`; IREE `SetEncoding`/`MaterializeEncoding` data tiling | Weights are stored in the exact tiled layout the kernel reads. Costs compile time and file size, buys coalescing at every inference. |
| Layout-free buffer annotation | Relax `AttachAttrLayoutFreeBuffers` | Marks which buffers the compiler is free to re-lay-out, so pre-packing is safe by construction. |

## Precision and quantization

Hexir supports f32 and f64. Every checked-in example is f64, which for a GPU is
the wrong default by a factor of thirty-two on consumer silicon.

| Transform | Prior art | What it buys |
| --- | --- | --- |
| Automatic mixed precision | Relay/Relax `ToMixedPrecision`; XLA `BFloat16Propagation`, `FloatNormalization` | Demotes what is safe to f16/bf16 and keeps accumulation wide. |
| Contraction input demotion | IREE `DemoteContractionInputsToBF16` | Narrower operands, f32 accumulate — the standard inference tradeoff. |
| Integer quantization | TVM QNN (`qnn.requantize`, `qnn.dequantize`); Relay quantization; IREE `LinalgQuantizedMatmulToMatmul` | int8 weights and activations, with the requantize arithmetic expressed in the IR rather than hidden in a kernel. |
| Fuse dequantize into matmul | IREE `FuseDequantizationMatmul`; MLC's fused dequant GEMV | Weights stay compressed in memory and are expanded in registers. The single biggest decode win — see [LLM serving](llm.md). |
| Narrow index and accumulator types | TIR `NarrowDataType` | Smaller address arithmetic, more registers for real work. |

## Placement and partitioning

Hexir's `hexir-partition` is the same idea as TVM's BYOC flow and IREE's device
affinity analysis, but with a static registry instead of a cost model, and with
no transfer insertion at all.

| Transform | Prior art | What it buys |
| --- | --- | --- |
| Annotate, merge, partition | Relay `AnnotateTarget` → `MergeCompilerRegions` → `PartitionGraph` | Merging is the missing middle step: adjacent same-device ops become one region, so placement is decided for subgraphs rather than one op at a time. |
| Composite pattern matching | Relay `MergeComposite`; IREE's `RaiseSpecialOps` | Recognize a multi-op pattern (softmax, attention, GELU written out longhand) and replace it with one op that has a good kernel. |
| Transfer insertion | IREE `stream.async.transfer` with affinity analysis; XLA `StreamAssignment` | Required before heterogeneous placement can ever be profitable. Hexir currently errors instead: a dispatch whose device disagrees with the active device is rejected in the VM. |
| Transfer cost in the objective | IREE's affinity/topology analysis | Placement that accounts for PCIe. Without it, a "faster" op can lose to the copy that feeds it. |
| Dynamic-to-static | Relay `DynamicToStatic` | Freeze shapes wherever they are actually known, so the static path stays fast even in a dynamic model. |

## Op coverage as an optimization

Not a transform, but it gates several: a pattern that cannot be expressed
cannot be fused.

`MulOp` is commented out in `HexirOps.td`, so there is no way to write a gated
activation. `softmax`, `gelu`, `tanh`, `sigmoid` and friends are declared with
no lowering. Missing entirely: batched matmul, `concat`, `slice`, reductions,
`embedding`/gather, `argmax`. TVM and IREE both treat a rich op set plus a
generic lowering as cheaper than a large kernel library, and both provide a
generic path (`te.compute` / `linalg.generic`) so a new elementwise op costs
one pattern rather than one kernel.
