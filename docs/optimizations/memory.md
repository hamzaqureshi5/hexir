# Memory

Where buffers come from, how long they live, and who is allowed to write into
them. Hexir currently answers all three the same way: every intermediate is a
fresh allocation that lives until the program ends.

The serializer assigns a slot per value and emits one `ALLOC` per `call_tir`
result; the VM turns each `ALLOC` into a device allocation and frees everything
at the end. Nothing is reused, nothing is written in place, and constants are
re-uploaded on every invocation. For a static graph — which is all Hexir can
compile — every one of those is solvable at compile time.

## Allocation planning

| Transform | Prior art | What it buys |
| --- | --- | --- |
| Liveness-based buffer reuse | TIR `StorageRewrite`; XLA `BufferAssignment` with `HloOrdering` and interference; Relay `GraphPlanMemory` | Two values whose live ranges do not overlap share one allocation. Peak memory drops toward the graph's true working set. |
| Static allocation plan | Relax `StaticPlanBlockMemory`; IREE `PackAllocations` + `LayoutSlices` | One arena allocated at load; every slot becomes an offset. Removes the device allocator from the hot path entirely — `cuMemAlloc` serializes against the driver. |
| Constant packing | IREE `PackConstants` | All weights in one contiguous upload instead of one per constant. |
| Allocation hoisting out of loops | IREE `EmplaceAllocations`; TIR `PlanAndUpdateBufferAllocationLocation` | Once the VM has control flow, per-iteration allocation is the first thing to go. |
| Compact allocations to what is used | TIR `CompactBufferAllocation` | A staged tile is sized to the tile, not to the whole buffer. Matters once shared memory is in play. |
| Workspace allocation | Relax `AllocateWorkspace` | One scratch buffer sized for the worst kernel, reused by all of them. |
| Free at last use | Relax `KillAfterLastUse` | Bounded live set even without a full plan. |
| Slot capacity | — | `HEXIR_MAX_SLOTS` is 256 and a dispatch accepts at most four arguments. A transformer graph exceeds both; a real plan makes the first limit meaningless anyway. |

## Aliasing and in-place update

| Transform | Prior art | What it buys |
| --- | --- | --- |
| Copy-on-write materialization | IREE `MaterializeCopyOnWrite` then `ElideAsyncCopies` | Introduce copies only where a write would be observed, then delete the ones that turn out to be unnecessary. |
| Copy insertion from alias analysis | XLA `CopyInsertion` on `HloAliasAnalysis` | The rigorous version: prove where a buffer may be shared, insert exactly the copies needed. |
| In-place elementwise | XLA's in-place dynamic-update-slice fusion; IREE's `outs` operand tying | `relu(x)` writes into `x`. Halves both allocations and traffic for every activation. |
| Destination passing | Relax `CallTIRRewrite`; IREE dispatch `outs` | Hexir already has this at the kernel boundary — `hexir.call_tir`'s verifier enforces callee arity of `args + 1`. What is missing is letting the destination *be* one of the inputs. |
| Donated / aliased arguments | XLA input-output aliasing and PJRT donated buffers | A caller hands ownership of a buffer to the executable, which writes the result into it. This is how a KV cache is updated without a copy. |
| Read-only marking | IREE `RefineUsage` — constant, variable, transient, staging, external | Lets the compiler know which buffers may be shared and which must be private. Weights are constant and can be uploaded once. |

## Rematerialization and offload

Only interesting once memory is actually a constraint, which for long-context
inference it becomes quickly.

| Transform | Prior art | What it buys |
| --- | --- | --- |
| Rematerialization | XLA `HloRematerialization` | Recompute a cheap value instead of holding it, to fit a memory budget. |
| Memory space assignment | XLA `MemorySpaceAssignment` (alternate memory), `HostOffloader` | Places values in a faster or a larger space, and schedules the prefetch. The general form of "keep weights on device, stream the rest". |
| Host-visible and pinned staging | IREE HAL memory types; Hexir's own `HEXIR_MEMORY_HOST_VISIBLE`, currently unused | Pinned staging makes host-to-device copies asynchronous and roughly twice as fast. Hexir's HAL already has the enum; the CUDA backend ignores it and treats everything as device-local. |
| Unified memory | CUDA managed memory, exposed by all three HALs | A fallback for models that do not fit, not an optimization. |

## Physical layout

Layout appears at the graph level too — see [Graph transforms](graph.md) — but
some of it is purely a memory concern.

| Transform | Prior art | What it buys |
| --- | --- | --- |
| Strided and permuted memrefs | MLIR layout maps | Hexir forces `IdentityLayoutMap` during bufferization, so no view is ever non-contiguous. That is a deliberate simplification with a cost: every layout change becomes a copy. |
| Data tiling / encodings | IREE `SetEncoding` → `MaterializeEncoding` | The tensor's in-memory form is chosen by the target, not by the frontend. |
| Padding and alignment | TIR `storage_align`; IREE pad-and-distribute | Avoids bank conflicts on device and split cache lines on host. |
| Splat and fill descriptors | IREE `flow.tensor.splat`; XLA broadcast fusion | Hexir's serializer expands a splat constant elementwise into RODATA, so a 256×256 constant of one value occupies 512 KB on disk and a full upload at runtime. A fill command would make it eight bytes. |
| Section reuse in place | Hexir's mmap loader | Already right: sections are used directly from the mapping, so a large RODATA costs nothing to *load*. It is the upload that is unoptimized, not the load. |

## What to do first

For a static model, in order:

1. Liveness-based reuse plus a single arena. Removes per-dispatch allocation.
2. Upload weights once at load, not per invocation. Requires separating a load
   phase from an execute phase in the VM.
3. In-place elementwise, which needs an aliasing bit in the executable entry so
   the VM can legitimately pass one slot twice.
4. A fill descriptor for splat constants, which is a small change to the
   serializer and the command list.
