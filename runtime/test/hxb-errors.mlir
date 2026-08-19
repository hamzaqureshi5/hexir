// Failure paths for the artifact: what the serializer refuses to emit, and
// what the runtime refuses to load or run.
//
// REQUIRES: runtime

// An op with no kernel has no place in a host program, and must be reported
// rather than silently dropped or crashed on. hexir.sigmoid is declared in
// TableGen but has no lowering.
// RUN: not %hexir -emit=hxb -o %t.bad.hxb %s 2>&1 | FileCheck %s --check-prefix=NOKERNEL
// NOKERNEL: cannot serialize this op into a host program

// A file that is not a module is rejected by the loader. This .mlir file is a
// convenient non-module.
// RUN: not %hexir-run %s 2>&1 | FileCheck %s --check-prefix=NOTAMODULE
// NOTAMODULE: invalid module

// A module records the device each kernel was placed on. Running it on a
// different device would give a right answer from the wrong hardware, so it is
// refused.
// RUN: %hexir -emit=hxb -o %t.cuda.hxb -placement=hexir.linear=cuda %S/hxb-vs-jit.mlir
// RUN: not %hexir-run --quiet %t.cuda.hxb 2>&1 | FileCheck %s --check-prefix=MISPLACED
// MISPLACED: placed on cuda but the active device is cpu

// The CUDA backend is real: it allocates on the device, copies host to device
// and back, and reports the actual GPU. Gated because it needs a working
// driver, which a CPU-only machine will not have.
// RUN: %hexir-run --selftest --device=cuda 2>&1 | FileCheck %s --check-prefix=CUDAHAL
// CUDAHAL: device        : cuda
// CUDAHAL: hal roundtrip : ok

// A cuda module on a cuda device gets past placement, allocates and transfers,
// then stops at dispatch: EXECUTABLES holds a descriptor, not device code.
// RUN: not %hexir-run --quiet --device=cuda %t.cuda.hxb 2>&1 | FileCheck %s --check-prefix=NOCODE
// NOCODE: rather than device code

// The HAL works without any module at all.
// RUN: %hexir-run --selftest | FileCheck %s --check-prefix=SELFTEST
// SELFTEST: device        : cpu
// SELFTEST: hal roundtrip : ok

func.func @main() {
  %a = hexir.constant dense<[[3.0, 1.0], [2.0, 2.0]]> : tensor<2x2xf64>
  %s = hexir.sigmoid %a : tensor<2x2xf64>
  hexir.print %s : tensor<2x2xf64>
  return
}
