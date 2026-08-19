# -*- Python -*-
# Compiler test suite: everything that only needs the `hexir` binary.
#
#   lit -v compiler/test
#   cd build && make check-hexir-compiler

import os

lit_config.load_config(
    config,
    os.path.join(os.path.dirname(__file__), "..", "..", "lit.common.cfg.py"),
)

config.name = "Hexir :: compiler"
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.hexir_build_dir, "test", "compiler")

if not os.path.exists(config.hexir_binary):
    lit_config.fatal("hexir binary not found at %s -- build the project first" % config.hexir_binary)
