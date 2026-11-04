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
suite (including resident input/output lifecycle and reload checks), runs Compute
Sanitizer memcheck, then records 20 warmed benchmark samples.
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

## Resident accounting sessions

`--mode pipeline` runs a transaction-cost sensitivity sweep on fixed prices and
holdings. `--operations 1..256` selects K distinct rates spanning `[0, 0.01]`;
K=1 uses `0.01`, and the default is K=8. Every rate invokes the unchanged float64
accounting kernel. Each method returns **all K operations' three per-path arrays**
in flat `[operation][path]` order. The CPU comparison recomputes the same K calls;
it does not measure a CPU algorithm that caches gain and turnover across rates.

```bash
/tmp/aster-cuda-build/aster_cuda --mode pipeline --paths 65536 --intervals 12 \
  --operations 32 --block-size 128 --warmups 5 --repeats 15 \
  --output /tmp/aster-pipeline-001.json
```

| Pipeline field | Completed timing boundary |
| --- | --- |
| `copy_per_call_ms` | K operations, each uploading both inputs, running one kernel, downloading three output arrays, and synchronizing |
| `resident_session_ms` | One input upload, K kernels writing distinct retained slots, three bulk downloads covering every operation, and one synchronization |
| `resident_preloaded_ms` | Inputs uploaded and synchronized before timing; K kernels, all output downloads, and synchronization |
| `native_cpu_sequence_ms` | K prepared native calls and all output writes on one CPU thread |

The primary comparison is `resident_session_ms` against `copy_per_call_ms`.
Residency, batched output copies, and fewer synchronizations all contribute to the
change. Pipeline measurements use completed host durations and introduce no CUDA
event calls within these methods. The older benchmark's `kernel_event_ms` remains
a separate single-kernel measurement.

All input arrays, output arrays, device slots, and sample storage are allocated
before warmups and timed sessions. The copy-per-call method writes directly into
views of its contiguous all-operation host outputs; it performs no extra vector
copies. Host vectors are pageable, so completed host durations include any driver
transfer staging. The separate cold fields record runtime/device setup, host
preparation, device allocation, and the first complete single-operation call on
the selected shape. These are not a full process startup measurement.

Pipeline JSON uses `schema_version: 2`; smoke, correctness, and the older benchmark
retain version 1. It records exact `workload.rates`, the input hash, device/build
identity, all raw `pipeline_samples`, each sample's reproducibly shuffled
`method_order`, all-output comparisons, and per-method numeric sums and an ordered
FNV-1a hash of the final checked outputs. The hash covers P&L, then costs, then
turnover, with operation and path order preserved. Comparisons check every output
of every measured repeat outside timing, using the same tolerances as the older
modes. A zero CPU duration from clock quantization is retained; GPU completed
samples must be positive.

Every sample's `transfers` records actual successful `cudaMemcpyAsync` requests at
the copy helper. For N paths, T intervals, and K operations, define input bytes
`I = N * (2*T + 1) * 8` and output bytes `O = K * N * 3 * 8`:

| Method | H2D bytes / calls | D2H bytes / calls |
| --- | --- | --- |
| `copy_per_call` | `K*I` / `2*K` | `O` / `3*K` |
| `resident_session` | `I` / `2` | `O` / `3` |
| `resident_preloaded` | `0` / `0` inside timing | `O` / `3` |

These counters measure requested copy payloads, not physical PCIe traffic. Input
changes require an explicit upload even when dimensions stay the same. Upload
invalidates every old output; launch before upload, invalid rates/slots/shapes,
incomplete outputs, and repeated readback without a fresh launch are rejected
before the corresponding CUDA calls.

Default correctness mode exercises the resident paths, so the existing
`--sanitizer required` runner checks them under Compute Sanitizer too. Cases cover
K=1 and multiple distinct rates, literal outputs, shorts, cancellation, partial
blocks, empty batches, changed shapes, and same-shape changed-input reloads.
This standalone session benchmark excludes Python binding costs, neural training,
and complete hyperparameter-optimization runtime. It measures fixed-input reuse;
it does not establish gains for a workload that changes prices or holdings on
every call.

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
