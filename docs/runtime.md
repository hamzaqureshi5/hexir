# The runtime

The compiler can hand you a file instead of running your program. The runtime
loads that file. It is plain C, and it links no MLIR and no LLVM — that is the
entire point. Deploying a compiled program should not mean shipping the
compiler.

```bash
./build/hexir -emit=hxb -o model.hxb mine.mlir   # compiler, ~230 MB
./build/hexir-run model.hxb                      # runtime, ~18 KB
```

```mermaid
flowchart LR
    subgraph compiler ["hexir  (links MLIR + LLVM)"]
        A["kernel level IR"] --> B["serializer"]
    end
    B --> C[("model.hxb")]
    subgraph runtime ["hexir-run  (links neither)"]
        C --> D["loader<br/>mmap + validate"]
        D --> E["interpreter<br/>replay commands"]
        E --> F["HAL<br/>cpu / cuda"]
    end
```

## The file

A header, a table of sections, then the section contents:

```text
┌────────────────────────────────┐
│ header   magic, version, count │
├────────────────────────────────┤
│ section table                  │  kind, offset, size  x N
├────────────────────────────────┤
│ symbols       entry points     │
│ program       command list     │
│ rodata        weights          │
│ executables   kernel entries   │
└────────────────────────────────┘
```

Sections are found by absolute file offset, and the whole file is mapped with
`mmap`. Weights are read straight out of the mapping and never copied, so
loading a large model costs almost nothing. That single requirement is why
offsets are absolute and payloads are 8-byte aligned.

Offsets come off disk, so they are untrusted. Every one is bounds-checked
before a pointer is handed out — a section claiming to extend past the end of
the file is rejected, not followed.

## The program

The host program is a flat list of commands, not bytecode:

| Command | Meaning |
| --- | --- |
| `ALLOC` | make a zeroed buffer in a slot |
| `CONST` | bind a slot to constant data in `rodata` |
| `DISPATCH` | run a kernel on some slots |
| `PRINT` | print a slot as a matrix |
| `END` | done |

Buffers are referred to by *slot*, a small dense index the compiler assigns and
the runtime resolves to a real allocation.

There is no bytecode interpreter because there is nothing to interpret yet:
everything the compiler can currently produce is straight-line dataflow with no
control flow, so a list to replay is enough. Branches become necessary when
dynamic shapes or control flow arrive.

## The HAL

Everything device-specific sits behind one vtable, so the host program is
identical whether it ends up on a CPU or a GPU.

```c
hexir_device_create(kind, &device);
hexir_buffer_allocate(device, size, memory_kind, &buffer);
hexir_buffer_write(buffer, data, size);
hexir_buffer_read(buffer, data, size);
hexir_device_wait(device);
```

Buffers declare where they live — `HOST_LOCAL`, `DEVICE_LOCAL` or
`HOST_VISIBLE`. That distinction is where placement decisions show up at
runtime: a `DEVICE_LOCAL` buffer needs an explicit transfer, a `HOST_VISIBLE`
one does not.

Only the CPU backend is written. Adding CUDA means adding one vtable in
`runtime/src/hal/cuda/` and touching no caller. It should use the driver API
(`libcuda`) rather than the CUDA runtime API, so it can be loaded dynamically
and a build without a toolkit still works.

## Placement is checked

A module records which device each kernel was compiled for. Running it
somewhere else is refused:

```text
hexir-run: kernel 'linear_0' is placed on cuda but the active device is cpu
```

Without that check a module built for the GPU would run quietly on the CPU and
print the right numbers, which is exactly the illusion this project exists to
avoid.

:::{important}
**The `executables` section holds kernel descriptions, not machine code.**

Each entry says *what* to compute — matrix multiply, these extents, this device
— and the runtime supplies the body from `reference_kernels.c`. That was the
way to make the file format, the loader, the HAL and the command list real and
testable end to end.

So a `.hxb` file today is portable, and its numbers are correct and match the
JIT exactly, but it is **not yet carrying compiled code**. Putting a CUBIN from
`gpu.binary` in that section, or a host object file, replaces the descriptions
without changing the file format or the command list. That is the next step.
:::

## Commands

```text
hexir-run <module.hxb> [--device=cpu|cuda] [--entry=name] [--quiet]
hexir-run --selftest
```

`--quiet`
: Print only the program's own output. Useful for diffing against the JIT.

`--selftest`
: Exercise the HAL — allocate, write, read back — without needing a module.

`--entry=name`
: Choose an entry point. Defaults to `main`.
