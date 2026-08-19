//===- reference_kernels.h - Stand-in kernel bodies --------------*- C -*-===//
//
// Reference implementations used while EXECUTABLES carries kernel descriptors
// rather than machine code (see the note on hexir_kernel_kind_t in program.h).
// Replacing a descriptor with an embedded CUBIN or host object removes the
// need for these without changing the container or the command list.
//
//===----------------------------------------------------------------------===//
#ifndef HEXIR_RUNTIME_REFERENCE_KERNELS_H
#define HEXIR_RUNTIME_REFERENCE_KERNELS_H

#include <stddef.h>
#include <stdint.h>

void hexir_ref_matmul_f64(const double *a, const double *b, double *c,
                          uint32_t m, uint32_t n, uint32_t k);
void hexir_ref_add_f64(const double *a, const double *b, double *c, size_t n);
void hexir_ref_relu_f64(const double *x, double *y, size_t n);

#endif
