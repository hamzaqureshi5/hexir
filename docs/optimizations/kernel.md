# Kernel transforms

Everything that happens to a loop nest. In Hexir this is the `hextir` level,
and the dialect is already shaped for it: `hextir.for` carries a `kind`
attribute that *is* the schedule, and `hextir.block` names a schedulable
region.

What is missing is the ability to change that schedule. `HexirToTIR.cpp` builds
one fixed nest per op, and `HexTIRToGPU.cpp` honours only `serial`, `parallel`
and `thread_binding` — `vectorized` and `unrolled` are accepted by the dialect
and silently lowered as ordinary loops.

## Where the current matmul stands

Worth being concrete about the baseline, because it sets the size of the prize.
`buildMatmulBody` gives one thread per output element with a serial reduction
and a register accumulator. Per multiply-add it issues two global loads. No
tiling, no shared memory, no reuse across threads, no vector loads, no tensor
cores. That is roughly one-twentieth of cuBLAS on an A6000, and every transform
below is a step from there.

The block size is chosen by divisibility (`chooseBlockSize`) because the kernel
has no bounds guard, so a dimension of 37 falls back to **one thread per
block**. A guard plus a ceiling-divided grid is a prerequisite for everything
here, not an optimization.

## Schedule primitives

TVM's TensorIR schedule is the most direct model for what `hextir` should be
able to express. Each primitive is a rewrite on the loop nest, and each has an
obvious `hextir` spelling.

| Primitive | TVM | IREE / XLA equivalent | Why Hexir wants it |
| --- | --- | --- | --- |
| `split` / `fuse` / `reorder` | `sch.split`, `sch.fuse`, `sch.reorder` | tiling levels in `iree_codegen.lowering_config` | The basis of everything else. `chooseBlockSize` is a hand-rolled `split` that cannot be re-parameterized. |
| `tile` | composition of split and reorder | IREE `LLVMGPUTileAndFuse` | Blocked matmul. The single highest-value change to the GPU path. |
| `bind` | `sch.bind` | workgroup distribution | Already present, as `thread_binding` plus the `thread` attribute. |
| `vectorize` | `sch.vectorize`, TIR `VectorizeLoop` | IREE `LLVMCPUVectorization`, vector distribution | 128-bit loads. The `kind` value already exists and does nothing. |
| `unroll` | `sch.unroll`, TIR `UnrollLoop` | — | Frees the reduction loop from address arithmetic. Also already a `kind` value that does nothing. |
| `cache_read` / `cache_write` | `sch.cache_read`, `sch.cache_write` | `GPUPromoteMatmulOperands` | Stages a tile in shared memory or registers. This is *the* prerequisite for a competitive matmul, and it needs `hextir.alloc_buffer` to carry a memory space. |
| `compute_at` / `reverse_compute_at` | `sch.compute_at` | fusion inside a dispatch region | Moves a producer inside the consumer's loop, which is how graph-level fusion actually becomes one kernel. |
| `compute_inline` | `sch.compute_inline`, `reverse_compute_inline` | IREE producer cloning | Recompute a cheap elementwise value instead of storing it. |
| `rfactor` | `sch.rfactor` | XLA `ReduceScatter`-style split reduction; IREE `LLVMCPUSplitReduction` | Splits a reduction so it can be parallel. This is what turns a decode GEMV from serial-over-K into split-K. |
| `decompose_reduction` | `sch.decompose_reduction` | — | Separates initialization from accumulation, so the init can be hoisted or vectorized separately. |
| `storage_align` | `sch.storage_align` | IREE `GPUReduceBankConflicts` | Pads a shared-memory tile to avoid bank conflicts. Cheap, and worth several percent. |
| `blockize` / `tensorize` | `sch.blockize`, `sch.tensorize` | IREE ukernels, `iree_gpu.multi_mma` | Replaces a matched sub-nest with a hardware intrinsic — MMA, dot-product, or a hand-written microkernel. |
| `transform_layout` | `sch.transform_layout` | IREE encodings | Changes a buffer's physical layout as a scheduling decision rather than a graph rewrite. |
| `pad_einsum` | `sch.pad_einsum` | IREE `LLVMCPUPeel`, pad-and-vector-distribute | Pads a contraction to a tile multiple, which removes the divisibility trap described above. |
| `rolling_buffer` | `sch.rolling_buffer` | — | Circular buffering for sliding-window computations. |
| `double_buffer` / async copy | TIR `InjectDoubleBuffer`, `InjectPTXAsyncCopy` | IREE `GPUMultiBuffering` + `GPUPipelining` | Overlaps the next tile's load with this tile's math using `cp.async`. Standard in every modern GEMM. |

## GPU-specific lowering

These are the passes that sit between a scheduled nest and correct device code.
Hexir currently needs none of them because its kernels are too simple to be
wrong in these ways — which is another way of saying each becomes required the
moment the corresponding schedule primitive lands.

| Pass | Prior art | Needed when |
| --- | --- | --- |
| Barrier insertion | TIR `ThreadSync` | Anything is staged in shared memory. |
| Cross-thread reduction | TIR `LowerCrossThreadReduction`; IREE `LLVMGPUWarpReduction` | A reduction is split across threads — warp shuffles and a final combine. Prerequisite for both flash attention and split-K GEMV. |
| Warp-level memory | TIR `LowerWarpMemory` | Values are exchanged through shuffles rather than shared memory. |
| MMA fragment inference | TIR `InferFragment`, `TransformMmaBufferLayout` | Tensor cores. Fragment layouts are not something to derive by hand. |
| Shared memory merging | TIR `MergeSharedMemoryAllocations` | Several staged buffers must fit one 48–228 KB budget. |
| Cooperative fetch | MetaSchedule `RewriteCooperativeFetch` | A tile load is distributed across the whole block, coalesced. |
| Occupancy and resource checks | MetaSchedule `VerifyGPUCode`; IREE `GPUCheckResourceUsage` | A schedule can now exceed registers or shared memory. Better a compile error than a launch failure. |
| Workgroup reordering | IREE `ReorderWorkgroups` | L2 locality between adjacent blocks. Small, free, and shape-dependent. |
| Loop partitioning | TIR `LoopPartition`; IREE `LLVMCPUPeel` | Bounds guards are hoisted out of the hot loop instead of being tested per iteration. |
| `if` hoisting, no-op removal, simplification | TIR `HoistIfThenElse`, `RemoveNoOp`, `Simplify` | The cleanup that makes generated nests survive to good code. |

## CPU-specific lowering

The CPU path is `convert-linalg-to-loops` with nothing before it, and the
runtime's fallback kernels in `reference_kernels.c` are scalar triple loops. For
a CPU-served static model this is the 10–50× range.

| Transform | Prior art | What it buys |
| --- | --- | --- |
| Tile, pack, vectorize | IREE `LLVMCPUTileAndFuse` + `LLVMCPUVectorization` | Cache blocking and SIMD. The MLIR passes exist; nothing adds them. |
| `mmt4d` / data tiling | IREE `linalg.mmt4d` with `SetEncoding` | Reformats operands into a tiled layout once, so the inner kernel is a dense register-blocked loop with no strided access. |
| Microkernels | IREE ukernels (`iree_uk_mmt4d`); TVM `tensorize` with CUTLASS or oneDNN | A hand-written inner loop where the compiler cannot win. Both projects concluded this is worth it for GEMM. |
| Thread the outer loop | TVM `sch.parallel`; OpenMP in the runtime kernels | The `parallel` loop kind already exists in `hextir` and lowers to a serial `scf.for`. |
| Fused multiply-add control | IREE `LLVMCPUUnfuseFMAOps` | Bit-exact agreement between the reference kernels and generated code — which the runtime tests depend on. |

## Search and autotuning

Once a schedule has parameters, someone has to choose them. All three systems
concluded that a fixed heuristic is not enough, and all three chose differently.

| Approach | System | Shape |
| --- | --- | --- |
| Template-based tuning | TVM AutoTVM | A human writes the schedule with knobs; the tuner picks the knobs. |
| Sketch generation | TVM Ansor / auto\_scheduler | The compiler derives the search space from the compute definition. |
| Rule-based space plus evolutionary search | TVM MetaSchedule — rules like `MultiLevelTiling`, `MultiLevelTilingTensorCore`, `AutoInline`, `CrossThreadReduction`, `AddRFactor`, `ParallelizeVectorizeUnroll`; mutators over tile sizes and thread bindings; an XGBoost cost model; results in a `JSONDatabase` | The most complete design, and the closest fit to a dialect that already stores schedules as attributes. |
| Default rules, no search | TVM Dlight — `gpu.Matmul`, `gpu.GEMV`, `gpu.Reduction`, `gpu.Transpose`, `gpu.Fallback` | Good schedules immediately, no tuning time. This is the pragmatic first step for Hexir: a small set of shape-classified default schedules. |
| Pipeline selection by pattern | IREE — a lowering config and translation info attached per dispatch, selecting `TileAndFuse`, `VectorDistribute`, `MatmulTensorCore`, `WarpReduction` | Deterministic, debuggable, no tuning database to ship. |
| Library and emitter selection, autotuned | XLA `GemmRewriter` + `GemmFusionAutotuner`, choosing between cuBLAS, cuDNN and Triton per fusion | Benchmark the candidates at compile time and cache the winner. |
| Scriptable schedules | MLIR transform dialect, as used by IREE | The schedule is a separate program applied to the IR. Useful for experiments without recompiling the compiler. |

Two practical notes. First, a tuning database must be keyed on
(op, shape, dtype, target) and shipped or cached, or compile time becomes
unusable — TVM ships records, XLA caches autotune results, IREE avoids the
problem by not searching. Second, `hextir`'s existing `annotate`-style
attributes are enough to record a chosen schedule, so a database could be
applied as an attribute-rewriting pass rather than a re-derivation.

## Attention and other special kernels

Some computations are not reachable by scheduling a naive nest; they need a
different algorithm.

| Kernel | Prior art | Note |
| --- | --- | --- |
| Flash / online attention | IREE `iree_linalg_ext.attention` and `online_attention`, decomposed via its aggregated-op interface; XLA's cuDNN fused-MHA rewrite; TVM via FlashInfer | Tiles the score matrix and keeps a running softmax, so the S×S intermediate is never materialized. Needs shared memory and cross-thread reduction first. |
| Softmax as a single kernel | XLA's Triton softmax emitter; Dlight `gpu.GeneralReduction` | Two passes over a row fused into one kernel with a running max. |
| Split-K / flash decoding | TVM `rfactor`; IREE `SplitReduction` | The decode-shaped GEMV case. See [LLM serving](llm.md). |
| `im2col` and Winograd | IREE `iree_linalg_ext.im2col`, `winograd.*` | Convolution strategies, if static vision models matter. |
| Sort, scan, top-k, argmax | IREE `iree_linalg_ext.{sort,scan,topk,arg_compare}` | Needed for on-device sampling, which is what removes a device-to-host round trip per token. |
