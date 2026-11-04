#!/usr/bin/env bash
# One sequential GPU owner. Never reuse an evidence directory.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: cuda/run_tests.sh --output-dir DIRECTORY [options]
  --output-dir DIR        New, exclusive directory for build and evidence
  --sanitizer auto|required|off   Compute Sanitizer policy (default: auto)
  --paths N              Benchmark paths (default: 65536)
  --intervals N          Benchmark intervals, 1..256 (default: 12)
  --block-size N         Threads per block (default: 128)
  --warmups N            Warmup calls (default: 5)
  --repeats N            Completed benchmark samples (default: 20)
  --device N             Visible CUDA device ordinal (default: 0)
  --help                 Show this message without needing CUDA

Requires Ubuntu/Linux with CUDA-devel (nvcc), CMake >=3.24, a C++20 host
compiler, Python 3, and nvidia-smi. Builds sm_80 for the rented A100.
EOF
}
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }
output_dir=''
sanitizer='auto'
paths=65536
intervals=12
block=128
warmups=5
repeats=20
device=0
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --output-dir|--sanitizer|--paths|--intervals|--block-size|--warmups|--repeats|--device)
            (($# >= 2)) || fail "missing value for $1"
            option=$1; value=$2; shift 2
            case "$option" in
                --output-dir) output_dir=$value ;;
                --sanitizer) sanitizer=$value ;;
                --paths) paths=$value ;;
                --intervals) intervals=$value ;;
                --block-size) block=$value ;;
                --warmups) warmups=$value ;;
                --repeats) repeats=$value ;;
                --device) device=$value ;;
            esac ;;
        *) fail "unknown option: $1" ;;
    esac
done
[[ -n "$output_dir" ]] || fail '--output-dir is required'
case "$sanitizer" in auto|required|off) ;; *) fail '--sanitizer must be auto, required, or off' ;; esac
for value in "$paths" "$intervals" "$block" "$warmups" "$repeats" "$device"; do
    [[ "$value" =~ ^[0-9]+$ ]] || fail 'numeric options require unsigned decimal integers'
done
# mkdir is the atomic no-overwrite gate, also rejecting files and dangling symlinks.
[[ ! -e "$output_dir" && ! -L "$output_dir" ]] || fail "output directory already exists: $output_dir"
mkdir -- "$output_dir" || fail "cannot create output directory: $output_dir"
output_dir=$(cd -- "$output_dir" && pwd -P)
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$script_dir/.." && pwd -P)
trap 'status=$?; if ((status != 0)); then printf "Run failed (exit %s); partial evidence retained in %s\n" "$status" "$output_dir" >&2; fi' EXIT
for program in nvcc cmake python3 nvidia-smi; do
    command -v "$program" >/dev/null 2>&1 || fail "required program is missing: $program (use a CUDA-devel image)"
done
if [[ "$sanitizer" == required ]] && ! command -v compute-sanitizer >/dev/null 2>&1; then
    fail 'Compute Sanitizer is required but compute-sanitizer is missing'
fi

uname -a > "$output_dir/host.txt"
if command -v lscpu >/dev/null 2>&1; then lscpu >> "$output_dir/host.txt"; fi
nvcc --version > "$output_dir/nvcc.txt"
cmake --version > "$output_dir/cmake.txt"
nvidia-smi -q > "$output_dir/nvidia-smi.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$output_dir/gpu-processes-before.csv"
python3 -I - "$repo_dir" "$output_dir/source-sha256.json" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
directories = ('native/include', 'native/src', 'native/tests', 'native/benchmarks', 'cuda/include', 'cuda/tests')
paths = [root / name for name in ('native/CMakeLists.txt', 'cuda/CMakeLists.txt', 'cuda/hedge_accounting.cu',
                                  'cuda/build_metadata.hpp.in', 'cuda/run_tests.sh')]
paths += [p for directory in directories for p in (root / directory).rglob('*')
          if p.is_file() and p.suffix in ('.cpp', '.hpp', '.cu', '.in', '.sh', '.py')]
paths = sorted(set(paths))
with open(sys.argv[2], 'x') as out:
    json.dump({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, out, indent=2)
    out.write('\n')
PY
cmake -S "$script_dir" -B "$output_dir/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTING=ON \
    > "$output_dir/configure.log" 2>&1
cmake --build "$output_dir/build" --parallel 2 --verbose > "$output_dir/build.log" 2>&1
binary="$output_dir/build/aster_cuda"
python3 -I - "$binary" "$output_dir/binary-sha256.txt" <<'PY'
import hashlib, pathlib, sys
with open(sys.argv[2], 'x') as out:
    out.write(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest() + '\n')
PY
# These checks include malformed CLI and no-overwrite behavior before device setup.
python3 -I "$script_dir/tests/check_cli.py" "$binary" > "$output_dir/cli-tests.log" 2>&1
"$output_dir/build/aster_cuda_comparison_test" > "$output_dir/comparison-tests.log" 2>&1
"$binary" --mode smoke --device "$device" --block-size "$block" --output "$output_dir/smoke.json"
"$binary" --mode correctness --device "$device" --block-size "$block" --output "$output_dir/correctness.json"
if [[ "$sanitizer" == off ]]; then
    printf '%s\n' 'Compute Sanitizer explicitly disabled (--sanitizer off).' | tee "$output_dir/sanitizer-status.txt"
elif command -v compute-sanitizer >/dev/null 2>&1; then
    printf '%s\n' 'Compute Sanitizer memcheck requested; see sanitizer.log and sanitizer.json.' > "$output_dir/sanitizer-status.txt"
    compute-sanitizer --tool memcheck --error-exitcode 99 --leak-check full \
        "$binary" --mode correctness --device "$device" --block-size "$block" \
        --output "$output_dir/sanitizer.json" > "$output_dir/sanitizer.log" 2>&1
    printf '%s\n' 'PASSED: Compute Sanitizer memcheck completed with exit 0.' > "$output_dir/sanitizer-status.txt"
else
    printf '%s\n' 'NOT RUN: compute-sanitizer is missing. Use --sanitizer required to make this fatal.' | tee "$output_dir/sanitizer-status.txt"
fi
"$binary" --mode benchmark --device "$device" --paths "$paths" --intervals "$intervals" \
    --block-size "$block" --warmups "$warmups" --repeats "$repeats" --output "$output_dir/benchmark.json"
python3 -I "$script_dir/tests/check_results.py" "$output_dir" > "$output_dir/result-checks.log" 2>&1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$output_dir/gpu-processes-after.csv"
printf 'Completed. Evidence: %s\n' "$output_dir"
