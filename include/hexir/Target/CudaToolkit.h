//===- CudaToolkit.h - Find a usable CUDA toolkit --------------*- C++ -*-===//
#ifndef HEXIR_TARGET_CUDATOOLKIT_H
#define HEXIR_TARGET_CUDATOOLKIT_H

#include <string>

namespace mlir {
namespace hexir {

/// A directory laid out the way MLIR expects when it looks for libdevice, i.e.
/// one where `<result>/nvvm/libdevice/libdevice.10.bc` exists.
///
/// Honours CUDA_ROOT / CUDA_HOME / CUDA_PATH first. Otherwise it finds
/// libdevice wherever the distribution put it and, if that layout is not the
/// one MLIR insists on, builds a directory of symlinks that is. Empty when no
/// libdevice can be found at all.
std::string resolveCudaToolkitPath();

} // namespace hexir
} // namespace mlir

#endif // HEXIR_TARGET_CUDATOOLKIT_H
