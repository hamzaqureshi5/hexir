# LLM serving

An autoregressive model is not just a bigger static model. It runs the same
weights thousands of times with a changing shape, it carries state between
invocations, and its two phases have opposite bottlenecks. Several entries here
are prerequisites rather than optimizations: without them the workload cannot be
expressed at all.

The prior art is TVM's MLC-LLM stack, IREE's sharktank/shortfin path, XLA as
used by JAX inference servers, and the serving systems — vLLM, SGLang,
TensorRT-LLM — which are not compilers but decided most of these questions
first.

## Two phases, two bottlenecks

| | Prefill | Decode |
| --- | --- | --- |
| Shape | `M` = sequence length, large | `M` = 1 (or batch size) |
| Bound by | compute | memory bandwidth on weights |
| Wants | tensor cores, tiling, large tiles | split-K reductions, quantized weights, no launch overhead |
| Kernel | GEMM | GEMV |

Hexir's matmul maps the `M` axis to `blockIdx.y`, so at `M = 1` the grid is one
block deep and a single row of threads each walks the whole `K` dimension
serially. Decode would run at a few percent of memory bandwidth. **Prefill and
decode genuinely want different kernels, selected on shape** — this is what
TVM's Dlight does with separate `gpu.Matmul` and `gpu.GEMV` rules, and what
TensorRT-LLM does with separate GEMM plugins.

## Blockers

None of these is an optimization. Each is a capability the current design
excludes.

| Blocker | Where it bites | Prior art for the fix |
| --- | --- | --- |
| Shapes are compile-time constants | `hexir_executable_entry_t` stores `m`, `n`, `k` and the launch geometry as `uint32_t`. Sequence length is not known when the module is written. | Symbolic shapes in Relax; IREE's dynamic dispatch dimensions and workgroup-count regions; XLA `DynamicPadder`. |
| No control flow in the VM | The command list is flat by design. A decode loop cannot be expressed. | Relax VM bytecode; IREE's VM. |
| No persistent state across invocations | `vm_release` frees every slot when the program ends. A KV cache must survive. | XLA donated buffers; IREE `util.global` variables and parameter archives. |
| No shared memory or cross-thread reduction in `hextir` | Flash attention and split-K both need them. | TIR `cache_read`, `ThreadSync`, `LowerCrossThreadReduction`. |
| No narrow or quantized types | `elem_size` accepts 4 or 8 bytes only. Decode is bandwidth-bound on weights, so this is the main lever. | TVM QNN and MLC's group quantization; IREE's quantized matmul path. |
| The op set cannot express a transformer | No `mul` (so no gated activation), no batched matmul, no `concat`, `slice`, reduction, gather, or `argmax`. | Any of the three. |

## Attention

| Technique | Prior art | Note |
| --- | --- | --- |
| Fused attention | IREE `iree_linalg_ext.attention` / `online_attention`; XLA's cuDNN fused-MHA rewriter; FlashInfer via TVM BYOC | Tiled scores with a running softmax. Avoids materializing the S×S matrix, which for long context is the difference between running and running out of memory. |
| Flash decoding / split-KV | vLLM and FlashInfer decode kernels; `rfactor` in TVM terms | At `M = 1` there is not enough parallelism in the query dimension, so the KV dimension has to be split and reduced. |
| Paged KV cache | vLLM PagedAttention; MLC's `PagedKVCache` TIR kernels | Fixed-size blocks with a block table, instead of one contiguous buffer per sequence. Removes the fragmentation that otherwise caps batch size. |
| Prefix / prompt caching | SGLang RadixAttention; vLLM prefix caching | Shared prefixes computed once across requests. A serving concern, but it constrains the cache layout. |
| Sliding window and sink attention | Serving systems | Bounded cache for long conversations. |
| Grouped-query attention | Model architectures, but it changes kernel shapes | Fewer KV heads than Q heads — the reduction and the cache layout both change. |
| Quantized KV cache | fp8 and int8 KV in TensorRT-LLM and vLLM | The cache eventually outgrows the weights; halving it doubles the context or the batch. |
| Chunked prefill | vLLM, SGLang | Splits a long prompt so it interleaves with decode work instead of blocking it. Needs control flow. |

## Weights

Decode reads every weight once per token and does almost no arithmetic per byte.
That makes weight bandwidth the whole problem, and compression the whole answer.

| Technique | Prior art | What it buys |
| --- | --- | --- |
| Weight-only int4 / int8 group quantization | MLC's `q4f16_1`; AWQ; GPTQ; Marlin kernels | Roughly 4× less weight traffic per token, with f16 arithmetic. The largest single decode speedup available. |
| Dequantize fused into the GEMV | IREE `FuseDequantizationMatmul`; MLC's fused dequant GEMV; Dlight's decode-GEMV rule | Weights stay packed in memory and expand in registers. Unfused dequantization writes the full f16 tensor back to memory and loses the entire benefit. |
| Weight pre-packing at compile time | Relax `LiftTransformParams`; IREE data tiling | The packed, tiled, possibly interleaved layout the kernel wants, computed once during compilation. |
| Weight residency | XLA donated buffers; IREE parameter archives | Upload once per process. Hexir re-uploads every constant on every `hexir_execute` — see [Memory](memory.md). |
| fp8 weights and activations | TensorRT-LLM, XLA on Hopper | Where the hardware supports it, another halving. |
| QKV and gate/up fusion | Relay `CombineParallelDense`; IREE `FuseSiluHorizontalMatmul` | Three or two projections over the same input become one wider matmul: one launch, one pass over the input. |

## Fusing the glue

Everything between the matmuls is bandwidth-bound, and there is a lot of it: two
normalizations, two residual adds, a RoPE, and a gated activation per layer.
Unfused, each is a full read and write of the activation tensor.

| Fusion | Prior art |
| --- | --- |
| RMSNorm with the residual add | Every serving stack; Dlight has an RMSNorm rule |
| RoPE into the QKV projection epilogue | TensorRT-LLM, vLLM |
| SwiGLU — `silu(gate) * up` — into the up-projection | IREE `FuseSiluHorizontalMatmul` |
| Residual add into the following normalization | XLA fusion, automatically |
| Sampling on device — argmax, top-k, top-p | IREE `iree_linalg_ext.{topk,arg_compare}` |

The last one is worth its own note: sampling on the host means a
device-to-host copy and a full synchronization *per token*, which on a small
model can rival the model itself.

## Reducing launches

A 32-layer model with seven kernels per layer is 224 launches per token before
any fusion. At `M = 1` each kernel may take less time than its launch.

| Technique | Prior art | Note |
| --- | --- | --- |
| CUDA graph capture of the decode step | XLA `CommandBufferScheduling`; Relax `RewriteCUDAGraph`; vLLM's captured decode graphs | The standard fix, and it is worth capturing per batch size. |
| Horizontal fusion of small kernels | XLA `HorizontalLoopFusion` | Independent small ops merged into one launch. |
| Persistent / mega-kernel decode | Research and recent serving work | One launch per token for the whole model. The endgame; only meaningful after everything else. |
| Layer code shared across layers | Relax `LambdaLift`, IREE VM functions | Also a compile-time and artifact-size argument: emitting 32 copies of identical kernels is what Hexir would do today. |

## Multi-GPU

Beyond the current scope, but it is the same partitioning problem Hexir already
models per op — so worth knowing where it leads.

| Technique | Prior art |
| --- | --- |
| Tensor parallelism | TVM Disco; XLA GSPMD sharding annotations |
| Collective fusion and overlap | XLA `AllReduceCombiner`, `CollectivePipeliner`, `AsyncCollectiveCreator` |
| Pipeline parallelism | XLA and serving systems |
| Sharding propagation | XLA `SpmdPartitioner` — annotate a few tensors, infer the rest |

## The shortest useful path

If the goal is an LLM running acceptably rather than a complete compiler, the
dependency chain is short and strictly ordered:

1. **Narrow types** — f16/bf16 activations. Everything else assumes them.
2. **Dynamic shapes in the artifact format**, with launch geometry computed at
   dispatch. Unblocks all of prefill.
3. **Control flow in the VM plus persistent buffers.** Unblocks decode and the
   KV cache.
4. **Weight residency and an allocation plan** — see [Memory](memory.md).
5. **A GEMV kernel with split-K**, and shape-based selection between it and the
   GEMM path.
6. **Weight-only int4 with fused dequantization.**
7. **Fused attention**, once shared memory and cross-thread reduction exist.
8. **Graph capture** of the decode step.

Steps 1–4 are format and runtime work; 5–8 are the kernel work from
[Kernel transforms](kernel.md). Neither half is useful without the other.
