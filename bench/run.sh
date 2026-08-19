#!/usr/bin/env bash
# Sweep square matmuls through hexir and cuBLAS and print a summary table.
#
#   bench/run.sh                                  # default sweep on the GPU
#   bench/run.sh --sizes=256,512,1024 --iters=20
#   bench/run.sh --device=cpu                     # the CPU path, no baseline
#
# Each size is compiled to a .hxb and run by hexir-bench, which times cuBLAS
# doing the same work and checks both against a CPU reference.

set -euo pipefail

sizes="128,256,512,1024,2048,4096"
iters=10
device="cuda"
build_dir=""

for arg in "$@"; do
  case "$arg" in
    --sizes=*)  sizes="${arg#*=}" ;;
    --iters=*)  iters="${arg#*=}" ;;
    --device=*) device="${arg#*=}" ;;
    --build=*)  build_dir="${arg#*=}" ;;
    -h|--help)  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# Default to <repo>/build, so the script works from anywhere in the tree.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -n "$build_dir" ] || build_dir="$repo_root/build"

hexir="$build_dir/hexir"
bench="$build_dir/hexir-bench"
for tool in "$hexir" "$bench"; do
  if [ ! -x "$tool" ]; then
    echo "not found: $tool" >&2
    echo "build first:  cmake -S . -B build && cmake --build build" >&2
    exit 1
  fi
done

# Everything a run produces sits under bench/work: the generated .mlir, the
# compiled modules and the full reports. Gitignored.
work="$repo_root/bench/work"
mkdir -p "$work"

# The compiler only places ops it knows about, and only the matmul matters here.
placement=""
[ "$device" = "cuda" ] && placement="-placement=hexir.linear=cuda"

printf '%-8s %14s %14s %14s %10s\n' size hexir cuBLAS ratio verify
printf '%-8s %14s %14s %14s %10s\n' ---- ----- ------ ----- ------

for n in ${sizes//,/ }; do
  src="$work/gemm-$n.mlir"
  mod="$work/gemm-$n.hxb"

  # No hexir.print: a printed 1024x1024 result is a million lines of stdout and
  # would dominate the measurement. Splat constants keep the source file small.
  cat > "$src" <<EOF
func.func @main() {
  %a = hexir.constant dense<1.000000e+00> : tensor<${n}x${n}xf64>
  %b = hexir.constant dense<2.000000e+00> : tensor<${n}x${n}xf64>
  %m = hexir.linear %a, %b : tensor<${n}x${n}xf64>
  return
}
EOF

  if ! "$hexir" -emit=hxb -o "$mod" $placement "$src" >/dev/null 2>"$work/compile-$n.log"; then
    printf '%-8s %14s\n' "$n" "compile failed"
    sed 's/^/    /' "$work/compile-$n.log" >&2
    continue
  fi

  out="$work/bench-$n.log"
  if ! "$bench" "$mod" "--device=$device" "--iters=$iters" >"$out" 2>&1; then
    printf '%-8s %14s\n' "$n" "run failed"
    sed 's/^/    /' "$out" >&2
    continue
  fi

  # Pull the numbers back out of the report rather than reformatting inside the
  # C tool, so the tool stays readable on its own.
  # Anchor on the colon: "^hexir " also matches the "hexir is 2.5x slower" line.
  hx=$(awk -F'[[:space:]]+' '/^hexir[[:space:]]*:/{print $5}' "$out")
  cb=$(awk -F'[[:space:]]+' '/^cuBLAS[[:space:]]*:/{print $5}' "$out")
  ratio=$(awk '/^hexir is /{print $3" "$4}' "$out")
  ok=$(awk '/^verify[[:space:]]*:/{print ($NF=="(ok)")?"ok":"FAILED"}' "$out")

  printf '%-8s %11s GF/s %11s GF/s %14s %10s\n' \
    "${n}x${n}" "${hx:--}" "${cb:--}" "${ratio:--}" "${ok:--}"
done

echo
echo "full reports in $work/bench-<size>.log"
