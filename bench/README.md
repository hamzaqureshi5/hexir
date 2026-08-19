# Benchmarks

`hexir-bench` runs a compiled `.hxb` module and times cuBLAS doing the same
work on the same device, with a correctness check against a CPU reference.
Without a baseline a GFLOPS figure means nothing, so all three are always
reported together.

## The script

`bench/run.sh` compiles each size and runs it, then prints a table:

```bash
bench/run.sh                                    # default sweep on the GPU
bench/run.sh --sizes=256,512,1024 --iters=20
bench/run.sh --device=cpu                       # the CPU path, no baseline
bench/run.sh --build=/path/to/build             # a build tree elsewhere
```

```text
size              hexir         cuBLAS          ratio     verify
----              -----         ------          -----     ------
128x128        16.93 GF/s      100.31 GF/s    5.9x slower         ok
256x256        35.97 GF/s      100.25 GF/s    2.8x slower         ok
512x512        54.76 GF/s      140.52 GF/s    2.6x slower         ok
1024x1024      61.01 GF/s      164.34 GF/s    2.7x slower         ok
```

Generated sources, modules and full reports land in `bench/work/`, which is
gitignored.

## The tool underneath

```bash
hexir -emit=hxb -o /tmp/g512.hxb -placement=hexir.linear=cuda bench/gemm/gemm-512.mlir
hexir-bench /tmp/g512.hxb --iters=10
```

```text
kernel   : linear_0  matmul  built for cuda  512x512x512  elem=8B
hexir    :    4.898 ms      54.80 GFLOP/s
cuBLAS   :    1.964 ms     136.68 GFLOP/s   (kernel only)
hexir is 2.5x slower than cuBLAS
verify   : max relative error 0 vs the CPU reference (ok)
```

Options: `--device=cpu|cuda`, `--iters=N`, `--no-baseline`, `--no-verify`.

## Reading the numbers

The two timings measure different things and the tool says so. The hexir figure
is the whole module — allocation, host-to-device copies, the kernel, the
synchronise — because that is what a caller pays. The cuBLAS figure is the
kernel alone. At 512x512 the transfers are about 6 MB, so a meaningful slice of
that 4.9 ms is PCIe rather than compute.

cuBLAS is dlopened, not linked, so this builds and runs on a machine without
CUDA; the baseline is skipped instead.

## Where the time goes on a GTX 1660 Ti

The card is TU116: no tensor cores, and **FP64 runs at 1/32 the FP32 rate**.
Everything in the dialect is `F64Tensor` today, so both sides are paying that.
Roughly 170 GFLOP/s is the f64 ceiling, which puts cuBLAS at about 80% of peak
and hexir at about 32%.

Moving the dialect to f32 is worth more than any scheduling work: it is a 32x
hardware factor, and it is the only path to the 5+ TFLOP/s the card can
actually reach.

## What to optimise, in order

1. **f32** — 32x, hardware, before any compiler work
2. **Coalescing** — check that `threadIdx.x` maps to the fastest-varying
   dimension so neighbouring threads touch neighbouring addresses
3. **Shared-memory tiling** — classic 32x32 tiles; Turing has 64 KB per SM
4. **Register blocking** — each thread computes a 4x4 or 8x8 micro-tile
5. **Autotuning** — search tile sizes for this specific card

Steps 2 to 4 are transformations over `hextir.for`, which already carries its
kind as an attribute. That is the substrate a scheduling language needs.

## The honest target

Beating cuBLAS at plain GEMM is not realistic; it is hand-tuned SASS per
architecture. Where a compiler wins is **fusion** — `linear + bias + relu` as
one kernel avoids two round trips to memory that a library cannot avoid — and
**shape specialisation**, since M, N and K are known at compile time here and
generic to cuBLAS.
