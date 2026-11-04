# A100 accounting correctness and timing package

This standalone C++20/CUDA executable compares a float64, path-major GPU kernel
with the checked native CPU library on identical arrays. Each path includes the
opening trade, every rebalance, and terminal liquidation. All three outputs
(P&L, costs, turnover) must satisfy `abs(actual - expected) <= 1e-12 +
1e-10 * abs(expected)`; a mismatch, nonfinite result, or CUDA error exits nonzero.

Local Apple hardware cannot compile or run this CUDA package. NVIDIA compilation,
GPU correctness, and Compute Sanitizer results require execution on the rented
A100. The host-only checks below are separate from GPU validation.

The [recorded A100 run](../docs/evidence/a100-2026-11-04/README.md) compiled and passed
the CUDA suite and required memcheck on actual hardware. Raw records and limitations
are preserved there; the small timing sweep does not establish whole-objective speedup.

## Run on the rented Ubuntu instance

Use an existing CUDA **devel** image containing `nvcc` (CUDA 12.6 or newer), a
compatible C++20 GCC host compiler, CMake 3.24+, Python 3, and `nvidia-smi`.
Ubuntu 24.04 supplies a sufficiently recent CMake. A runtime-only CUDA image is
insufficient. Install host tools if needed, then run from the repository checkout:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake python3
nvcc --version
nvidia-smi
bash cuda/run_tests.sh --output-dir /tmp/aster-a100-run-001 --sanitizer required
```

The result directory must not exist; its parent must exist. A failed run retains
partial evidence. Choose a new directory for a rerun. No Python packages, Python
extension, network service, or additional GPU instance are required.

The runner builds Release for `sm_80`, checks CLI/no-overwrite behavior and the
numerical comparator, runs hand-checked smoke cases and the complete correctness
suite, runs Compute Sanitizer memcheck, then records 20 warmed benchmark samples.
Only one GPU-owning process runs at a time. Device ordinal `0` means the first
visible device; the process inventory is saved so other tenants/processes remain
visible when interpreting timings. This does not establish an otherwise idle GPU.

`--sanitizer auto` is the default: it runs memcheck when available and explicitly
records a missing tool. `--sanitizer required` makes absence fatal;
`--sanitizer off` explicitly records that it was disabled. Instrumented timings
in `sanitizer.json` are correctness evidence, not performance measurements.

For a different fixed workload, select it directly; this package does not perform
a launch tuning study:

```bash
bash cuda/run_tests.sh --output-dir /tmp/aster-a100-run-002 \
  --paths 2048 --intervals 12 --block-size 128 --warmups 5 --repeats 30 \
  --device 0 --sanitizer required
```

## Build and inspect individual runs

```bash
cmake -S cuda -B /tmp/aster-cuda-build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build /tmp/aster-cuda-build --parallel 2
ctest --test-dir /tmp/aster-cuda-build --output-on-failure
/tmp/aster-cuda-build/aster_cuda --mode correctness --output /tmp/aster-check-001.json
/tmp/aster-cuda-build/aster_cuda --mode benchmark --paths 256 --intervals 12 \
  --block-size 128 --warmups 5 --repeats 20 --output /tmp/aster-bench-001.json
```

`--mode smoke` is the default; omit `--output` to emit JSON on standard output.
An output file is reserved exclusively before GPU initialization and is never
overwritten. On failure it may remain empty; only JSON with `"passed": true`
constitutes a successful report. `--help` and invalid CLI arguments need no GPU.

Correctness includes literal results `0.089 / 0.011 / 1.1`, shorts, one interval,
zero holdings/costs, an empty batch, partial blocks, 1–256 intervals, and exact or
near cancellation in both gross gain and final P&L. Random prices are in
`[0.5, 2]`, holdings in `[-1, 1]`, and cost rates in `[0, 0.01]`. Every benchmark
repeat compares every output against the same CPU reference outside timing.
The kernel uses one thread per path and preserves chronological summation.
CUDA FMA contraction and fast arithmetic are disabled; the host uses
`-ffp-contract=off -fno-fast-math`. Do not add conflicting compiler flags.

## Read the evidence

`benchmark.json` retains each completed sample; no averages are relabeled as tail
latency, and no whole-objective speedup is inferred.

| Field | Measured boundary |
| --- | --- |
| `kernel_event_ms` | Kernel only, CUDA events in the same stream, read after completion |
| `completed_host_ms` | Host-to-device copies, kernel, all three device-to-host copies, stream synchronization |
| `native_cpu_prepared_ms` | One-thread native arithmetic and output writes on prevalidated inputs; allocation/validation excluded |
| `cold.runtime_setup_ms` | First runtime/device initialization and hardware queries |
| `cold.device_allocation_ms` | Reusable device buffers, stream, and events |
| `cold.first_completed_host_ms` | First small smoke call, including transfers and first kernel launch |

Device buffers and events are reused. Host arrays are pageable; host timing
includes any transfer staging. JSON construction, correctness comparison, input
creation, allocation, and checksums are outside warmed samples. The cold fields
are separate stages, not a full process startup measurement. Cold smoke and warm
benchmark have different shapes.

The exclusive evidence directory also contains device/driver/runtime and compiler
metadata, process inventory, complete build commands, source and binary SHA-256
hashes, build logs, and explicit sanitizer status. JSON contains the input hash
(FNV-1a over host-native input bytes and rate), seed, shape, launch settings,
checksums, errors, and raw durations. Retain this directory when sharing results.
Source files do not contain remote access details or credentials.

## Host-only checks

These need neither CUDA nor the optional Python extension:

```bash
python3 -m unittest discover -s cuda/tests -v
mkdir -p cuda/.build-local
c++ -std=c++20 -Icuda/include cuda/tests/test_comparison.cpp \
  -o cuda/.build-local/test_comparison
cuda/.build-local/test_comparison
bash -n cuda/run_tests.sh
```

The comparator regression test deliberately corrupts each output, checks nonfinite
values and near-zero tolerance, and must reject those cases. The shell tests run
the real argument/output boundary and verify existing evidence is preserved.

References: [NVIDIA asynchronous execution and events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html),
[NVIDIA Compute Sanitizer and nonzero error exit codes](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).
