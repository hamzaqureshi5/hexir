# Command line

## hexir

```text
hexir [--compile-to=<phase>] [-emit=<format>] [-o <file>] [-opt]
      [-placement=<op>=<device>,...] [input.mlir]
```

With no input file, the compiler builds a small program in C++
(`compiler/Support/Builder.cpp`) and compiles that instead.

### --compile-to

How far to compile. Defaults to `artifact`.

| Phase | Output |
| --- | --- |
| `input` | the `hexir` graph, as written, with no passes run |
| `graph` | after graph optimization |
| `place` | `device` on every op, `hexir.targets` on the module |
| `kernel` | each compute op as a `hextir` kernel |
| `device` | CUDA kernels as a `gpu.module`, lowered to NVVM |
| `artifact` | a loadable module (see `-o`) |

`device` stops at NVVM rather than going on to a binary. That is the IR `ptxas`
is handed, and it means the device path can be read on a machine with no CUDA
toolkit installed.

These three are a **frozen route**: `hexir` → linalg → bufferize → LLVM.
Nothing has executed it since the JIT was removed, and it bypasses the kernel
level, so no scheduling work applies to it. It is kept as a dump until host code
generation from `hextir` replaces it.

| Phase | Output |
| --- | --- |
| `linalg` | linalg on tensors, carrying placement |
| `linalg-gpu` | CUDA-placed linalg ops as `gpu.launch` |
| `linalg-llvm` | the LLVM dialect |

### -emit

What to write out. A phase produces one kind of thing, so most of the time this
can be left off.

| Format | Meaning |
| --- | --- |
| `mlir` | textual MLIR. The default for every phase but `artifact` |
| `llvm` | translated LLVM IR. Needs `--compile-to=linalg-llvm` |
| `hxb` | the binary module. Needs `--compile-to=artifact`, and is its default |

A pair that cannot mean anything is refused with a reason rather than guessed
at, and the old single-flag spellings (`-emit=mlir-tir`, `-emit=mlir-hetero`,
and the rest) still work but warn and name their replacement.


### Other options

`-o <file>`
: Output path for `--compile-to=artifact`. Defaults to `out.hxb`.

`-opt`
: Run LLVM's O3 pipeline before executing or printing.

`-placement=<op>=<device>[,...]`
: Override where an operation runs. Keys are graph-level op names
  (`hexir.linear`, `hexir.add`, `hexir.relu`); devices are `cpu`, `cuda`, or
  `gpu` as an alias for `cuda`. An unsupported pair is rejected before any pass
  runs.

MLIR's own pass-manager flags work too, including `--print-ir-after-all` and
`--mlir-print-ir-after-failure`.

### Examples

```bash
hexir --compile-to=artifact -o built-in.hxb              # built-in program, CPU
hexir-run built-in.hxb                                   # run it
hexir --compile-to=kernel -placement=hexir.linear=cuda   # matmul as a GPU kernel
hexir --compile-to=device -placement=hexir.linear=cuda   # ... and its NVVM
hexir --compile-to=artifact -o model.hxb mine.mlir       # ship a file
hexir --compile-to=place --print-ir-after-all mine.mlir  # watch every pass
```

## hexir-run

```text
hexir-run <module.hxb> [--device=cpu|cuda] [--entry=name] [--quiet]
hexir-run --selftest
```

`--device=`
: Which device to run on. Defaults to `cpu`. `cuda` is not implemented yet.
  A module is refused if it was compiled for a different device.

`--entry=`
: Entry point name. Defaults to `main`.

`--quiet`
: Suppress the module description and print only program output.

`--selftest`
: Check the HAL without needing a module.

### Exit codes

`0` success · `1` load or execution failed · `2` bad arguments
