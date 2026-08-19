# How it works

A compiler is a stack of translations. Each level describes the same program,
but answers a different question about it.

```mermaid
flowchart TD
    G["Graph — hexir dialect<br/>what to compute"]
    P["Placement<br/>where each op runs"]
    K["Kernel — hextir dialect<br/>how one device computes it"]
    L["Loops and buffers — linalg, memref, scf<br/>the actual loop nests"]
    M["Machine — LLVM, NVVM<br/>instructions"]
    A[".hxb file<br/>something you can ship"]

    G --> P --> K --> L --> M
    K --> A

    style G fill:#e8f0fe,stroke:#4285f4
    style P fill:#fce8e6,stroke:#ea4335
    style K fill:#e6f4ea,stroke:#34a853
    style A fill:#fef7e0,stroke:#fbbc04
```

## The five levels, one paragraph each

**Graph.** Your program as whole-tensor operations: a matrix multiply, a relu,
a print. No loops, no memory, no devices. This is the `hexir` dialect.

**Placement.** A pass walks the graph and writes a `device` attribute on every
operation, either `"cpu"` or `"cuda"`. It reads its decisions from a small
registry (`TargetSupport`) that you can override from the command line. This
happens *before* any lowering, so the decision travels down with the op.

**Kernel.** Each operation becomes a small function over buffers, with explicit
loops. This is the `hextir` dialect. Placement stops being a note here and
becomes real: a CPU op gets `parallel` loops, a GPU op gets loops bound to
`blockIdx.x` and `threadIdx.x`.

**Loops and buffers.** Standard MLIR from here: `linalg` for structured ops,
`memref` for buffers, `scf` for loops. Hexir does not invent anything at this
level because MLIR already has a good one.

**Machine.** LLVM IR for the CPU, NVVM then PTX then CUBIN for the GPU.

## Why placement comes first

The interesting question in this compiler is *where* things run, so the answer
is computed early and carried everywhere.

```mermaid
flowchart LR
    A["hexir.linear<br/>device = cuda"] --> B["linalg.matmul<br/>device = cuda"]
    B --> C["gpu.launch<br/>device = cuda"]
```

Every lowering pass copies the `device` attribute onto whatever it creates. So
by the time you reach GPU code generation, the ops that should become GPU
kernels are already labelled, and no pass has to guess.

The cost of this design is that the attribute can get dropped: a pass that
rebuilds an op without copying it loses the decision. That is why the partition
pass runs a second time, as a safety net for anything unlabelled.

## Out to a file

Once each operation is a kernel, the compiler hands you a file.

`-emit=hxb` serializes the program into a module and stops. `hexir-run` loads
that file later; it links no MLIR and no LLVM, so it is tens of kilobytes
instead of hundreds of megabytes.

A GPU kernel is compiled to a CUBIN and embedded, so the runtime launches it
with no compiler present. The same program compiled for either device must
produce the same numbers, and there is a test that diffs them to make sure.

## What to read next

* [The two IRs](ir-levels.md) — what `hexir` and `hextir` actually look like
* [The pass pipeline](pipeline.md) — every pass, in order, and why
* [The runtime](runtime.md) — the file format and how it is executed
