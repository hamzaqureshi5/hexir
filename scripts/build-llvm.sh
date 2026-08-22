#!/usr/bin/env bash
# Build the pinned LLVM/MLIR in third_party/llvm-project.
#
# Hexir needs an unreleased LLVM: the new-style OpTy::create(builder, ...)
# builders are not in any release, so no packaged MLIR can build this project.
# The submodule pins the exact commit it is known to build against, and this
# script configures that commit the way hexir needs it.
#
# Everything lands in third_party/llvm-project/build. Re-running is
# incremental. Expect 30-90 minutes and roughly 10 GB the first time.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/third_party/llvm-project"
build="$src/build"
jobs="$(nproc)"
cuda=auto

usage() {
  cat <<'USAGE'
usage: scripts/build-llvm.sh [-j N] [--no-cuda] [-- <extra cmake args>]

  -j N         parallel build jobs (default: nproc)
  --no-cuda    skip the MLIR CUDA runner even if nvcc is present
USAGE
}

extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j) jobs="$2"; shift 2 ;;
    --no-cuda) cuda=off; shift ;;
    --) shift; extra=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "$src/llvm/CMakeLists.txt" ]]; then
  echo "error: $src is empty. Run:" >&2
  echo "    git submodule update --init third_party/llvm-project" >&2
  exit 1
fi

echo "== pinned commit"
git -C "$src" log -1 --format='   %h  %ad  %s' --date=short

args=(
  -S "$src/llvm" -B "$build" -G Ninja
  -DLLVM_ENABLE_PROJECTS=mlir
  # NVPTX is not optional: it is what makes a CUDA-placed kernel compilable.
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX"
  -DCMAKE_BUILD_TYPE=Release
  # Asserts on. MLIR without them turns a bad rewrite into a confusing crash
  # much later, and this is a compiler under development.
  -DLLVM_ENABLE_ASSERTIONS=ON
  # Linking MLIR is memory hungry; two at a time keeps an 8-16 GB box alive.
  -DLLVM_PARALLEL_LINK_JOBS=2
  -DLLVM_INCLUDE_BENCHMARKS=OFF
  -DLLVM_INCLUDE_EXAMPLES=OFF
)

command -v ccache >/dev/null && args+=(-DLLVM_CCACHE_BUILD=ON)
command -v lld    >/dev/null && args+=(-DLLVM_USE_LINKER=lld)

if [[ "$cuda" != off ]] && command -v nvcc >/dev/null; then
  # CMAKE_CUDA_COMPILER is not optional, and getting it wrong costs an hour.
  # MLIR's cuda runner check fires when the variable is unset, and
  # check_language(CUDA) CACHES its answer -- so a configure that ran before
  # the toolkit was installed leaves NOTFOUND behind forever. Passing it
  # explicitly sidesteps the cache. See docs/llvm-cuda-build.txt.
  args+=(-DMLIR_ENABLE_CUDA_RUNNER=ON "-DCMAKE_CUDA_COMPILER=$(command -v nvcc)")
  echo "== cuda: on ($(command -v nvcc))"
else
  echo "== cuda: off (no nvcc, or --no-cuda). The CPU path still works end to end."
fi

cmake "${args[@]}" "${extra[@]+"${extra[@]}"}"
ninja -C "$build" -j "$jobs"

echo
echo "== done. Configure hexir with:"
echo "   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
echo "   (it finds this build on its own)"
