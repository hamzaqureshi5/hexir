//===- CudaToolkit.cpp - Find a usable CUDA toolkit -----------------------===//
//
// MLIR looks for libdevice at exactly `<toolkit>/nvvm/libdevice/libdevice.10.bc`.
// NVIDIA's own installs have that layout; Ubuntu's nvidia-cuda-toolkit package
// does not -- it puts libdevice at `/usr/lib/nvidia-cuda-toolkit/libdevice/`
// with no `nvvm` directory at all. The result is
//
//   error: LibDevice path: /usr/nvvm/libdevice/libdevice.10.bc does not exist
//
// which reads like a broken CUDA install and is not one.
//
// So: find libdevice wherever it actually is, and if the surrounding directory
// does not have the layout MLIR insists on, build a small directory of symlinks
// that does. Cheap, done once, and it means a plain `apt install
// nvidia-cuda-toolkit` is enough to compile for the GPU.
//
//===----------------------------------------------------------------------===//

#include "hexir/Target/CudaToolkit.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <cstdlib>

using namespace llvm;

namespace {

bool hasLibDevice(StringRef toolkit) {
  SmallString<256> path(toolkit);
  sys::path::append(path, "nvvm", "libdevice", "libdevice.10.bc");
  return sys::fs::is_regular_file(path);
}

/// Directories that hold libdevice, in the layout each packaging uses.
/// Returns the directory *containing* libdevice.10.bc.
std::string findLibDeviceDir() {
  static const char *candidates[] = {
      "/usr/lib/nvidia-cuda-toolkit/libdevice", // Ubuntu / Debian
      "/usr/local/cuda/nvvm/libdevice",         // NVIDIA installer
      "/usr/lib/cuda/nvvm/libdevice",
      "/opt/cuda/nvvm/libdevice",
  };
  for (const char *dir : candidates) {
    SmallString<256> path(dir);
    sys::path::append(path, "libdevice.10.bc");
    if (sys::fs::is_regular_file(path))
      return std::string(dir);
  }
  return {};
}

/// Build `<cache>/nvvm/libdevice -> <real>` so MLIR's fixed lookup succeeds.
std::string buildShim(StringRef libdeviceDir) {
  SmallString<256> cache;
  if (!sys::path::cache_directory(cache) && !sys::path::home_directory(cache))
    return {};
  sys::path::append(cache, "hexir", "cuda-root");

  SmallString<256> nvvm(cache);
  sys::path::append(nvvm, "nvvm");
  if (sys::fs::create_directories(nvvm))
    return {};

  SmallString<256> link(nvvm);
  sys::path::append(link, "libdevice");
  if (!sys::fs::exists(link))
    // Failure here is not fatal: the check below decides.
    (void)sys::fs::create_link(libdeviceDir, link);

  std::string root(cache);
  return hasLibDevice(root) ? root : std::string();
}

} // namespace

std::string mlir::hexir::resolveCudaToolkitPath() {
  // An explicit setting always wins, even if it turns out not to work -- being
  // overridden and then silently ignored is worse than a clear failure.
  for (const char *var : {"CUDA_ROOT", "CUDA_HOME", "CUDA_PATH"})
    if (const char *value = std::getenv(var))
      if (*value)
        return std::string(value);

  std::string libdeviceDir = findLibDeviceDir();
  if (libdeviceDir.empty())
    return {};

  // If the parent already has the expected layout, use it directly.
  StringRef parent = sys::path::parent_path(libdeviceDir);
  if (sys::path::filename(parent) == "nvvm") {
    std::string root(sys::path::parent_path(parent));
    if (hasLibDevice(root))
      return root;
  }
  return buildShim(libdeviceDir);
}
