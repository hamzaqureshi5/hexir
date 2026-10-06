# ninja

A small MLIR dialect for learning TableGen and ODS (Operation Definition
Specification). Each op in `include/ninja/NinjaOps.td` shows one idea.

**Start with [docs/ods-guide.md](docs/ods-guide.md)**: every ODS concept in simple terms, plus
a list of exercises that cover them all.

| Op | Teaches |
| --- | --- |
| `ninja.constant` | attributes vs. results, declarative assembly format |
| `ninja.add` | operands, type constraints, traits |
| `ninja.strike` | a custom C++ verifier (`hasVerifier`) |

## Layout

```
include/ninja/NinjaOps.td      the op definitions -- start here
include/ninja/NinjaDialect.h   includes the generated declarations
lib/NinjaDialect.cpp           includes the generated definitions + hand-written C++
tools/ninja-opt/               mlir-opt with the ninja dialect registered
test/                          lit + FileCheck tests
```

`mlir-tblgen` turns `NinjaOps.td` into `build/include/ninja/*.inc`. Reading
those files is the fastest way to see what ODS writes for you.

## Build and run

Needs an MLIR build (see `third_party/README.md`, or pass `-DMLIR_DIR=...`).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/ninja-opt test/ninja-roundtrip.mlir
./build/ninja-opt test/ninja-roundtrip.mlir --mlir-print-op-generic
./build/ninja-opt test/ninja-invalid.mlir

cmake --build build --target check-ninja   # needs `pip install lit`
```
