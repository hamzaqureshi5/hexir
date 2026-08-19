# -*- Python -*-
# Runtime test suite: everything that runs `hexir-run`.
#
# These are the artifact tests. They compile with `hexir` too, because the
# thing under test is whether the runtime can execute what the compiler
# produced -- a test lives with the binary whose behaviour it asserts, and for
# these that is the runtime.
#
#   lit -v runtime/test
#   cd build && make check-hexir-runtime

import os

lit_config.load_config(
    config,
    os.path.join(os.path.dirname(__file__), "..", "..", "lit.common.cfg.py"),
)

config.name = "Hexir :: runtime"
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.hexir_build_dir, "test", "runtime")

if not os.path.exists(config.hexir_run_binary):
    lit_config.fatal(
        "hexir-run not found at %s -- build with -DHEXIR_BUILD_RUNTIME=ON" % config.hexir_run_binary
    )
