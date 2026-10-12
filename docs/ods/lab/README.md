# Lab sources

Every *(lab op)* example in this guide comes from one of these three small projects. They are copies
of the ninja project with extra ops added, kept so you can read the full code behind an example,
rebuild it, and run it yourself.

| Folder | Contains | Used in chapters |
|---|---|---|
| [ops/](ops/) | `dodge`, `shadow`, `sum`, `mix`, `stars`, `dojo`/`yield`, folding, canonicalization, DRR (`NinjaPatterns.td`), the `stance` enum, the `!ninja.shuriken` type, the `#ninja.rank` attribute, the `ninja-count-strikes` pass (`Passes.td`) | 4, 5, 9, 10, 12, 13, 14 |
| [attributes/](attributes/) | every example op and drill answer from [practice-attributes.md](../practice-attributes.md) (`AttrLab.td`, `DrillLab.td`) | practice page |
| [traits/](traits/) | one op per trait in [traits/include/ninja/TraitsLab.td](traits/include/ninja/TraitsLab.td), plus the interface-trait ops and their C++ | 6, 7, 8, 11 |

Each folder is a complete CMake project. Build it **outside** the folder, so no build files end
up in `docs/`:

```bash
cmake -S docs/ods/lab/traits -B /tmp/ninja-traits-lab -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ninja-traits-lab -j
/tmp/ninja-traits-lab/ninja-opt docs/ods/lab/traits/examples/algebra.mlir --canonicalize
```

## Example inputs

| File | Try with |
|---|---|
| `ops/examples/attributes-formats-regions.mlir` | `--mlir-print-op-generic` |
| `ops/examples/fold-canonicalize.mlir` | `--canonicalize`, `--ninja-count-strikes=verbose=true --mlir-disable-threading` |
| `ops/examples/add-zero.mlir` | `--canonicalize` |
| `ops/examples/drr.mlir` | `--canonicalize` |
| `ops/examples/shuriken-type.mlir` | (no flags) |
| `ops/examples/custom-stars.mlir` | `--mlir-print-op-generic` |
| `ops/examples/rank-attr.mlir` | `--mlir-print-op-generic` |
| `traits/examples/algebra.mlir` | `--canonicalize` |
| `traits/examples/memory-effects.mlir` | `--cse`, then `--canonicalize` |
| `traits/examples/speculation.mlir` | `--loop-invariant-code-motion --allow-unregistered-dialect` |
| `traits/examples/region-verifier.mlir` | (no flags). Reports one error **on purpose**: the empty `ninja.arena` |

The lab ops are not part of your ninja dialect. Add them to your own ninja through the exercises.
