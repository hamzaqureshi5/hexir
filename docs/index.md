# Hexir

Hexir is a small compiler for neural networks. You give it a graph of
operations, it decides which ones run on the CPU and which on the GPU, and it
turns them into code you can run.

It is built on [MLIR](https://mlir.llvm.org/), and it is small enough to read.
That is the point: the gap between MLIR's Toy tutorial and a production
compiler like IREE is enormous, and Hexir sits in the middle.

```mermaid
flowchart LR
    A["your program<br/><i>tensors</i>"] --> B["decide<br/>where each op runs"]
    B --> C["turn each op<br/>into a kernel"]
    C --> D["run it now<br/><i>JIT</i>"]
    C --> E["write a file<br/><i>.hxb</i>"]
    E --> F["run it later<br/><i>hexir-run</i>"]
```

Two ways to run the same program:

* **Now** — compile and execute in one process, the usual way to develop.
* **Later** — write a `.hxb` file and run it with `hexir-run`, a small program
  that contains no compiler at all. A GPU kernel is compiled to a CUBIN and
  embedded, so the runtime launches it with no compiler present.

## Start here

If you want to *use* it, read [Getting started](getting-started.md).

If you want to understand *how it works*, read [How it works](overview.md)
first. It is a picture and eight paragraphs. Then pick whichever of the
detailed pages you need.

```{toctree}
:maxdepth: 2
:caption: Using Hexir

getting-started
```

```{toctree}
:maxdepth: 2
:caption: How it works

overview
ir-levels
pipeline
runtime
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/cli
reference/dialects
reference/layout
cuda-server
```

## Where it stands

Hexir is research software. Some parts are finished and some are scaffolding,
and the docs say which is which rather than leaving you to find out.

| Works | Partly | Not yet |
| --- | --- | --- |
| CPU path, end to end | GPU kernels are one block, one thread | Transfer insertion in the JIT path |
| Per-operation placement | CPU kernels in `.hxb` are descriptions, not code | Memory planning |
| `.hxb` artifacts, CPU **and GPU** | | More operations, and a frontend |
