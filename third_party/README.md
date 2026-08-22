# third_party

## llvm-project

A submodule pinned to one commit, because "which LLVM?" is not a detail in this
project. Hexir uses the new-style `OpTy::create(builder, ...)` MLIR builders,
which are in no released LLVM, so no packaged MLIR can build it. Before this
submodule existed, the answer to "which commit does hexir need?" was whatever
happened to be checked out on the machine.

| | |
| --- | --- |
| Pinned commit | `6e493ed169c4` (`llvmorg-24-init-5403-g6e493ed169c4`) |
| Version it reports | `24.0.0git` |
| Remote | `https://github.com/hamzaqureshi5/llvm-project`, a fork of `llvm/llvm-project` |

The commit is an ordinary upstream commit; the fork is just the URL it is
fetched from. Nothing hexir needs is fork-only, so the pin can be moved to
upstream by editing `.gitmodules` if that is ever preferable.

### Getting it

```bash
git submodule update --init third_party/llvm-project   # ~3 GB
scripts/build-llvm.sh                                  # 30-90 min, ~10 GB
```

A blobless clone is much smaller if you do not need LLVM's history:

```bash
git clone --filter=blob:none https://github.com/hamzaqureshi5/llvm-project
```

Hexir's own `cmake` finds this build on its own once it exists. The order it
looks in is `-DLLVM_DIR=` on the command line, then this submodule, then a
machine-wide checkout. It prints which one it chose, because pointing `LLVM_DIR`
at a directory with no config in it does *not* fail — CMake quietly searches its
default prefixes instead, and there can easily be three LLVMs on one machine.

### Moving the pin

```bash
cd third_party/llvm-project
git fetch origin && git checkout <new-sha>
cd ../.. && git add third_party/llvm-project && git commit
```

Then rebuild, run `make check-hexir`, and update the commit named in this file.
If the major version changes, update `HEXIR_LLVM_REQUIRED_MAJOR` in the root
`CMakeLists.txt` too — it is what turns an API break into one line of
explanation instead of pages of template errors.

### Two things to keep true

**The runtime must not see any of this.** `runtime/` is standalone C99 with no
MLIR and no LLVM, and that is the point: deploying a compiled module must not
mean shipping the compiler. Nothing under `runtime/` may reference this
directory. Its workflow is currently commented out in
`.github/workflows/runtime.yml`; when it comes back, leave `actions/checkout` on
its default of *not* fetching submodules, or a suite that needs no LLVM will
start downloading 3 GB of it.

**The submodule is not a place to keep patches.** If hexir ever does need an
unlanded LLVM change, put it on a branch of the fork that is only ever
fast-forwarded deliberately, and say here what the patch is and why. A pin that
silently drifts is worse than no pin.
