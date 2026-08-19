# -*- Python -*-
# Shared lit configuration.
#
# Loaded by compiler/test/lit.cfg.py and runtime/test/lit.cfg.py, which set
# `config.name` and their own required tools. Everything to do with finding the
# build directory and the tools lives here so the two suites cannot drift.

import os
import shutil

import lit.formats

config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir", ".test"]

# Locate the build directory: --param build_dir=..., $HEXIR_BUILD_DIR, or
# <repo>/build.
_repo_root = os.path.dirname(os.path.abspath(__file__))
build_dir = lit_config.params.get(
    "build_dir",
    os.environ.get("HEXIR_BUILD_DIR", os.path.join(_repo_root, "build")),
)
build_dir = os.path.abspath(build_dir)


def find_tool(names, extra_dirs=()):
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

hexir = os.path.join(build_dir, "hexir")
hexir_run = os.path.join(build_dir, "hexir-run")

# load_config() runs this file in its own namespace, so anything the per-suite
# configs need has to be hung off `config`.
config.hexir_build_dir = build_dir
config.hexir_binary = hexir
config.hexir_run_binary = hexir_run

# %hexir-run must be registered BEFORE %hexir: lit applies substitutions in
# order, so the shorter pattern would otherwise eat the prefix of the longer.
if os.path.exists(hexir_run):
    config.substitutions.append(("%hexir-run", hexir_run))
    config.available_features.add("runtime")
if os.path.exists(hexir):
    config.substitutions.append(("%hexir", hexir))
config.substitutions.append(("FileCheck", filecheck))

# MLIR's NVVM target finds libdevice through these, and Ubuntu's toolkit does
# not use NVIDIA's directory layout, so a shim path is often needed. Pass them
# through rather than letting tests fail on a missing libdevice.
for var in ("CUDA_ROOT", "CUDA_HOME", "CUDA_PATH"):
    if var in os.environ:
        config.environment[var] = os.environ[var]

# Tests needing the CUDA toolkit are gated `REQUIRES: cuda`.
if shutil.which("nvcc"):
    config.available_features.add("cuda")
