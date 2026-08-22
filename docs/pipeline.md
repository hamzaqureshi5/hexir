# The pass pipeline

All pass ordering lives in `compiler/Pipelines/Pipelines.cpp`, one function per
phase, called from `buildHexirPipeline`. It adds passes and returns as soon as it
reaches the phase you asked for, and the driver runs the whole thing once.

```mermaid
flowchart TD
    S["canonicalize, shape inference, CSE<br/>graph"]
    S --> P["partition<br/>place"]
    P --> T{"which route?"}

    T -->|kernel, device, artifact| K2["lower-to-tir"]
    K2 --> KD{"device?"}
    KD -->|"kernel, artifact"| KE(["stop"])
    KD -->|device| KG["hextir to gpu.module<br/>then NVVM"]

    T -->|"linalg, frozen"| L2["lower-to-linalg"]
    L2 --> L3["partition again<br/>safety net"]
    L3 --> L6["bufferize<br/>tensors become memrefs"]
    L6 --> L7["cuda ops to gpu.launch"]
    L7 --> L8["linalg to loops"]
    L8 --> L9["outline kernels, NVVM, CUBIN"]
    L9 --> L10["scf to cf, then LLVM"]
```

## Why the order matters

`Phase` is only *partly* ordered, and knowing which parts is the difference
between a working stage and one that prints the wrong IR.

`input → graph → place` is a genuine sequence, and the pipeline gates it with
`>=`. Below `place` it **forks**, and choosing a route is a branch, not a
comparison.

That distinction is not pedantry. The old `Stage` enum claimed to be one ordered
list gated entirely with `>=`, and an affine stage ended up sitting below the
linalg stage as a result: asking for it returned before adding any pass, so it
printed the graph unchanged and looked like a stage that simply had nothing
interesting to show. Both it and the pass behind it have since been deleted.

Two consequences worth knowing:

* `artifact` stops at the kernel level rather than running the device passes. It
  has to: the command list is built from kernel-level IR, so `ModuleSerializer`
  compiles device code on a *clone* and leaves this module alone.
* `device` stops at NVVM rather than going on to a binary. That is the IR
  `ptxas` is handed, and it means the device path can be inspected — and tested
  — on a machine with no CUDA toolkit.

## The passes

### Cleanup

`canonicalize` → `hexir-shape-inference` → `canonicalize` → `cse`.

Shape inference fills in result shapes through an interface each op
implements. It is nested on `hexir::FuncOp`, which only exists if the program
was built with `hexir.func` ops.

### Placement

`hexir-partition` writes `device = "cpu"` or `device = "cuda"` on each op,
reading from the `TargetSupport` registry in `compiler/Target/TargetInfo.cpp`. The
`-placement` flag overrides that registry before any pass runs.

It runs **before** lowering, so the decision can travel with the op. It skips
anything that already has a `device` attribute, which is what lets a second run
act as a safety net without overwriting real decisions.

### Kernel level

`hexir-lower-to-tir` turns each compute op into a `hextir.prim_func` plus a
`hexir.call_tir`, choosing loop kinds from the `device` attribute. Terminal —
nothing lowers `hextir` further yet.

### Graph to loops

`hexir-lower-to-linalg` rewrites `hexir.linear` into `linalg.matmul`,
`hexir.relu` into a `linalg.generic` with `arith.maximumf`, and constants into
`arith.constant`. Every pattern copies the `device` attribute onto whatever it
creates.

### Placement is visible without a detour

`--compile-to=linalg` prints linalg ops carrying their `device` attribute. Mirror
`ls_cpu` / `ls_gpu` dialects used to exist so placement appeared in the op name;
they were removed because the round trip through them rebuilt each op and lost
its destination operand and attributes.

### Bufferization

Tensors become buffers. Three things had to be right here, and each was a bug
at some point:

1. **`empty-tensor-to-alloc-tensor` must run first.** `tensor.empty` has no
   bufferization of its own, and the `alloc_tensor` created mid-flight never
   enters the worklist, so it survives as an unbufferized op.
2. **Do not use a constant as a destination.** A destination-passing `outs`
   operand must be writable. An `arith.constant` is not, so bufferization has
   to emit a read-only global plus a copy. Use `tensor.empty` and
   `linalg.fill`.
3. **Every tensor op needs a `BufferizableOpInterface`**, and for most dialects
   that comes from an external model you must register. `hexir.print` gets one
   from `compiler/Support/BufferizableOpInterfaceImpl.cpp`.

When this goes wrong you get `error: op was not bufferized` with no location.
Run with `--mlir-print-ir-after-failure` and look for what still has tensor
operands; `to_tensor` and `to_buffer` are allowed in the output, anything else
is the culprit.

### GPU

For ops with `device == "cuda"`, `hexir-lower-cuda-to-gpu` builds a
`gpu.launch` by hand. Then the standard chain: outline the kernel into a
`gpu.module`, attach the NVPTX target, convert to NVVM, compile to CUBIN with
`ptxas`, and rewrite the host side into CUDA runtime calls.

:::{warning}
The generated `gpu.launch` uses a 2D launch geometry. The outer loop maps to
`blockIdx.y`, and the inner loop is split across blocks and threads (`blockIdx.x`
and `threadIdx.x`). The resulting grid and block extents travel through the
module descriptor to the HAL, which feeds them to `cuLaunchKernel`. While better
than a purely scalar loop, this is still a naive schedule without shared memory
or register tiling.
:::

### LLVM

`convert-linalg-to-loops` must run before GPU lowering finishes, because
`gpu-to-llvm` cannot handle live linalg ops. Then `scf` becomes control flow,
`hexir-to-llvm` lowers the remaining ops (including turning `hexir.print` into
`printf` calls), and unrealized casts are reconciled.
