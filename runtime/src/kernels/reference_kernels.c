#include "reference_kernels.h"

void hexir_ref_matmul_f64(const double *a, const double *b, double *c,
                          uint32_t m, uint32_t n, uint32_t k) {
  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (uint32_t p = 0; p < k; ++p)
        acc += a[(size_t)i * k + p] * b[(size_t)p * n + j];
      c[(size_t)i * n + j] = acc;
    }
  }
}

void hexir_ref_add_f64(const double *a, const double *b, double *c, size_t n) {
  for (size_t i = 0; i < n; ++i)
    c[i] = a[i] + b[i];
}

void hexir_ref_relu_f64(const double *x, double *y, size_t n) {
  for (size_t i = 0; i < n; ++i)
    y[i] = x[i] > 0.0 ? x[i] : 0.0;
}
