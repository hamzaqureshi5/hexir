# -*- Python -*-
# lit configuration for the ninja tests.
#
#   cd build && make check-ninja
#   lit -v test                          (from the repo root; finds ./build)
#   lit -v test --param build_dir=/path/to/build

import os
import shutil

import lit.formats

config.name = "Ninja"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)

repo_root = os.path.dirname(config.test_source_root)
build_dir = os.path.abspath(lit_config.params.get(
    "build_dir", os.environ.get("NINJA_BUILD_DIR", os.path.join(repo_root, "build"))))
config.test_exec_root = os.path.join(build_dir, "test")

ninja_opt = os.path.join(build_dir, "ninja-opt")
if not os.path.exists(ninja_opt):
    lit_config.fatal("ninja-opt not found at %s -- build the project first" % ninja_opt)
config.substitutions.append(("%ninja-opt", ninja_opt))

# FileCheck: PATH first, then the LLVM build tree, then distro packages.
filecheck = shutil.which("FileCheck")
for d in ["/home/user/llvm-project/build/bin",
          os.path.join(repo_root, "third_party/llvm-project/build/bin")] + \
         ["/usr/lib/llvm-%d/bin" % v for v in (22, 21, 20, 19, 18)]:
    if not filecheck and os.path.exists(os.path.join(d, "FileCheck")):
        filecheck = os.path.join(d, "FileCheck")
if not filecheck:
    lit_config.fatal("FileCheck not found")
config.substitutions.append(("FileCheck", filecheck))
