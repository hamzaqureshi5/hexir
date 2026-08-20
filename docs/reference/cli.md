# Command line

## hexir

```text
hexir [-emit=<stage>] [-o <file>] [-opt] [-placement=<op>=<device>,...] [input.mlir]
```

With no input file, the compiler builds a small program in C++
(`compiler/Support/Builder.cpp`) and compiles that instead.

### -emit

| Value | Output |
| --- | --- |
| `mlir` | the `hexir` graph, as written |
| `mlir-tir` | each compute op as a `hextir` kernel |
| `hxb` | a loadable module (see `-o`) |
| `mlir-affine` | *accepted, but no affine pass is wired up* |
| `mlir-linalg` | after lowering to linalg on tensors |
| `mlir-hetero` | linalg ops carrying their `device` attribute |
| `mlir-gpu` | CUDA-placed ops as `gpu.launch` |
| `mlir-llvm` | the LLVM dialect |
| `llvm` | translated LLVM IR |


### Other options

`-o <file>`
: Output path for `-emit=hxb`. Defaults to `out.hxb`.

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
hexir -emit=hxb -o built-in.hxb                       # built-in program, CPU
hexir-run built-in.hxb                                # run it
hexir -emit=mlir-tir -placement=hexir.linear=cuda     # matmul as a GPU kernel
hexir -emit=hxb -o model.hxb mine.mlir                # ship a file
hexir -emit=mlir-linalg --print-ir-after-all mine.mlir # watch every pass
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
