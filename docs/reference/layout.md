# Source layout

The project follows the layout MLIR and IREE use: public headers and TableGen
under `include/`, implementations mirrored under `lib/`, tools separate.

```text
include/hexir/
  Dialect/Hexir/IR/          dialect header, HexirOps.td
  Dialect/Hexir/Transforms/  passes that stay inside one dialect
  Dialect/HexTIR/IR/
  Dialect/LS/{IR,Transforms}/
  Conversion/Passes.h        passes that cross dialects
  Pipelines/Pipelines.h      pass ordering, and the Stage enum
  Serialization/             the .hxb writer
  Target/, Support/

lib/
  Dialect/<Name>/{IR,Transforms}/
  Conversion/<A>To<B>/       one directory per conversion
  Pipelines/  Serialization/  Target/  Support/

tools/hexir/                 the compiler driver
runtime/                     the standalone C runtime
test/                        lit + FileCheck
cmake/HexirLibrary.cmake     the hexir_library() helper
```

## Transforms or Conversion?

This is the split that matters.

**Transforms** — the pass rewrites a dialect into itself. Shape inference and
partitioning stay in `Dialect/Hexir/Transforms/`.

**Conversion** — the pass turns one dialect into another. It lives in
`Conversion/<A>To<B>/`, named for the pair: `HexirToLinalg`, `HexirToTIR`,
`HexirToLLVM`, `LSToLinalg`, `CudaToGpu`.

## Adding things

There is no glob. Each directory has a `CMakeLists.txt` calling
`hexir_library(...)`, which builds a static library, wires its TableGen
dependencies, and registers itself in a global property that `tools/hexir`
links.

So:

* **a new file** needs a line in its directory's `CMakeLists.txt`
* **a new directory** needs an `add_subdirectory` in its parent
* **a new `.td` file** needs an entry in the matching `include/.../IR/CMakeLists.txt`

That is deliberate. Which module a file belongs to should be a decision, not a
side effect of where it was saved.

## The runtime is separate

`runtime/` has its own `CMakeLists.txt`, builds standalone, and has **no
include path into `include/` and no MLIR or LLVM dependency**. It is added from
the root build *before* the LLVM include directories so it cannot inherit them
by accident.

If the runtime ever needs to link an MLIR library, the layering is wrong.
