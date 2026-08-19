# Execution

Everything between "the kernels are compiled" and "the answer is available".
This is the cheapest section of the catalogue to act on, because none of it
requires changing a single kernel — and on a small model, all of it dominates.

Hexir's runtime replays a flat command list. Each `DISPATCH` on the CUDA path
loads the CUBIN, looks up the entry point, launches, synchronizes the context,
and unloads the module. That is several milliseconds of driver work wrapped
around a kernel that may take microseconds.

## Launch overhead

| Transform | Prior art | What it buys |
| --- | --- | --- |
| Cache loaded modules and functions | Every runtime does this; IREE loads an executable once per device | Removing `cuModuleLoadData`/`cuModuleUnload` from the dispatch path. The key is the executable index, which outlives any one command. |
| Command buffers | IREE HAL `hal.command_buffer`, recorded once and reused; Vulkan's model | Record the whole command list once, submit it per invocation. The abstraction Hexir's HAL is missing. |
| CUDA graphs | XLA `CommandBufferScheduling`; Relax `RewriteCUDAGraph`; torch.compile's `cudagraph_trees` | The concrete version of the above on NVIDIA. Hexir's command list is static straight-line dataflow *by construction*, which is exactly the precondition graph capture wants. For decode, this is a large win. |
| Deduplicate executables | IREE `DeduplicateExecutables` | Identical kernels compiled once, loaded once. |
| Elide redundant commands | IREE `ElideRedundantCommands` | Removes repeated bindings and barriers from a recorded buffer. |
| Indirect / deferred dispatch | IREE indirect command buffers | Launch geometry read from a buffer, so a recorded command buffer survives a shape change. The bridge between graph capture and dynamic shapes. |
| Fold uniform operands, specialize dispatches | IREE `FoldUniformOperands`, `SpecializeDispatches` | Constant launch parameters become compile-time constants in the kernel. |

## Asynchrony

Hexir synchronizes twice per dispatch: `cuda_launch` calls
`cuCtxSynchronize` before returning, and the CPU path ends with a device wait.
A pipeline that drains after every operation cannot overlap anything.

| Transform | Prior art | What it buys |
| --- | --- | --- |
| Streams and a timeline | IREE `stream.timepoint` with semaphores; XLA's thunk streams | Launch everything, wait once at the point a result is actually read. |
| Schedule for concurrency | IREE `ScheduleExecution` and `ScheduleConcurrency` — partitions a DAG into waves | Independent branches of the graph run concurrently instead of in source order. |
| Propagate and elide timepoints | IREE `PropagateTimepoints`, `ElideTimepoints` | Removes waits that are already implied by another dependency. |
| Latency-hiding scheduler | XLA `LatencyHidingScheduler` | Reorders the schedule so long operations overlap with independent work. |
| Async copies | CUDA `cuMemcpyAsync` on a stream; IREE `stream.async.transfer` | Overlaps the weight upload with the first layers, and the result download with the last. |
| Overlap compute with collectives | XLA `AsyncCollectiveCreator`, `CollectivePipeliner` | Multi-GPU only, but the reason tensor parallelism scales at all. |

Two things in the current design make asynchrony easy to add: results are only
observed at `PRINT` and `END`, and `hexir_buffer_read` is the only path back to
the host. Those are natural sync points and there are no others.

## The virtual machine

`program.h` documents the choice honestly: a flat command list, because
everything the compiler can produce is straight-line dataflow, and "bytecode
becomes necessary when dynamic shapes or control flow arrive."

For LLMs, they have arrived.

| Capability | Prior art | Needed for |
| --- | --- | --- |
| Control flow | TVM's Relax VM bytecode; IREE's VM bytecode with `scf`-derived branches; XLA `while` | A decode loop, early exit on end-of-sequence, chunked prefill. |
| Dynamic shapes | IREE `ExpandTensorShapes` and dynamic dispatch dims; XLA `DynamicPadder`; Relay/Relax symbolic shapes | Sequence length. Hexir bakes `m`, `n`, `k` *and* the launch geometry into `hexir_executable_entry_t` as `uint32_t`, so this is a format change, not a pass. |
| Shape computation at runtime | Relax's symbolic shape expressions; IREE's workgroup count region on `hal.executable.export` | Grid size derived from a shape register instead of a constant. |
| Function calls and closures | Relax `LambdaLift`; IREE VM functions | Sharing one layer's code across thirty-two layers, rather than emitting it thirty-two times. |
| Reference counting or explicit lifetimes | IREE VM refs; Relax `KillAfterLastUse` | Bounded memory once control flow means lifetimes are not lexical. |
| Loop optimizations | XLA `WhileLoopInvariantCodeMotion`, `WhileLoopConstantSinking`, `WhileLoopTripCountAnnotator`; TIR `LoopPartition` | Hoisting anything shape-invariant out of the decode loop. |
| Separate load and execute phases | IREE parameter archives (IRPA), mmapped and bound once; Relax `LiftTransformParams` | Weights uploaded and preprocessed once per process, not once per call. Currently every `hexir_execute` re-runs every `CONST`. |

## Multi-device and heterogeneous execution

Hexir's premise is per-op placement, so this is closer to the project's centre
than the rest of the catalogue — but the machinery is not there yet: a dispatch
whose recorded device disagrees with the active device is rejected rather than
bridged.

| Transform | Prior art | What it buys |
| --- | --- | --- |
| Device affinity analysis | IREE's affinity attributes and analysis | Every value knows which device holds it, so transfers are inserted by the compiler rather than assumed by the programmer. |
| Transfer insertion and elision | IREE `stream.async.transfer` plus copy elision | The missing pass. Without it, heterogeneous placement is only expressible when it happens not to need a copy. |
| Queue and topology awareness | IREE device topology; XLA `StreamAssignment` | Placement that knows the interconnect, not just the op. |
| Collectives | IREE `flow.collective`, `stream` channels; XLA all-reduce combiners and splitters | Tensor-parallel inference across GPUs. |
| Multi-device partitioning | XLA GSPMD / `SpmdPartitioner`; TVM Disco | Sharding a model too large for one device. The far end of the roadmap, but it is the same partitioning problem Hexir already models per op. |

## Measurement

An optimization you cannot see is an optimization you cannot keep.

`bench/hexir-bench.c` times the first executable in the module against cuBLAS,
which is the right idea for a GEMM and blind to everything else. What it does
not report: per-dispatch breakdown, launch overhead as a share of wall time,
achieved bandwidth for memory-bound kernels, or a decode-shaped case where
`M = 1`.

For reference, the standard practice in the three projects:

* **Roofline framing** — report achieved bandwidth for memory-bound kernels and
  achieved FLOPS for compute-bound ones. GFLOPS alone hides the entire
  elementwise and decode story.
* **Per-dispatch tracing** — IREE's dispatch instrumentation and XLA's profiler
  attribute time to individual kernels, which is what makes fusion decisions
  arguable rather than aesthetic.
* **A tracked baseline** — cuBLAS for GEMM, and for the CPU path a BLAS or a
  ukernel, so a regression is visible as a ratio rather than as a number that
  looks fine in isolation.
