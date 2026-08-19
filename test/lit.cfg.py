# -*- Python -*-
# Lit configuration for the Hexir test suite.
#
# Standalone config: locates the hexir binary in the CMake build directory
# and FileCheck from the system LLVM. Run with either:
#   cd build && make check-hexir
#   lit -v test                      (from the repo root)
#   lit -v test --param build_dir=/path/to/build

import os
import shutil

import lit.formats

config.name = "Hexir"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir", ".test"]
config.test_source_root = os.path.dirname(__file__)

# Locate the build directory: --param build_dir=..., $HEXIR_BUILD_DIR,
# or default to <repo>/build.
build_dir = lit_config.params.get(
    "build_dir",
    os.environ.get(
        "HEXIR_BUILD_DIR",
        os.path.join(config.test_source_root, "..", "build"),
    ),
)
build_dir = os.path.abspath(build_dir)
config.test_exec_root = os.path.join(build_dir, "test")

hexir = os.path.join(build_dir, "hexir")
if not os.path.exists(hexir):
    lit_config.fatal(
        "hexir binary not found at %s — build the project first" % hexir
    )


def find_tool(names, extra_dirs):
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    for directory in extra_dirs:
        for name in names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    return None


filecheck = find_tool(
    ["FileCheck", "FileCheck-20", "FileCheck-19", "FileCheck-18", "FileCheck-14"],
    ["/usr/lib/llvm-20/bin", "/usr/lib/llvm-19/bin", "/usr/lib/llvm-18/bin",
     "/usr/lib/llvm-14/bin"],
)
if not filecheck:
    lit_config.fatal("FileCheck not found in PATH or /usr/lib/llvm-*/bin")

# %hexir-run must be registered BEFORE %hexir: lit applies substitutions in
# order, so the shorter pattern would otherwise eat the prefix of the longer.
hexir_run = os.path.join(build_dir, "hexir-run")
if os.path.exists(hexir_run):
    config.substitutions.append(("%hexir-run", hexir_run))
    config.available_features.add("runtime")

config.substitutions.append(("%hexir", hexir))
config.substitutions.append(("FileCheck", filecheck))

# Expose 'cuda' feature when nvcc (CUDA toolkit) is on PATH.
# Tests that require full GPU lowering (gpu-module-to-binary, jit) are guarded
# with REQUIRES: cuda so they are skipped on machines without the toolkit.
if shutil.which("nvcc"):
    config.available_features.add("cuda")
