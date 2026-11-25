# Testing GPU hedge accounting on an A100

This C++20/CUDA program runs the same hedging calculation on a GPU and a CPU.
Both use 64-bit floats with one row per price path. They calculate profit and
loss (P&L), transaction costs, and turnover from the opening trade through the
final trade that closes the position. Each GPU output must satisfy
`abs(actual - expected) <= 1e-12 + 1e-10 * abs(expected)` against the native CPU
result. The program exits with an error on a mismatch, a nonfinite value, or a
CUDA error.

The CUDA suite and Compute Sanitizer memcheck passed on an NVIDIA A100. The
timings in that run cover the accounting calculation. Building and testing the
GPU code requires a CUDA development host. On an Apple machine, use the
[host-only checks](#host-only-checks). [A100 test results](../docs/evidence/a100-2026-11-04/README.md)

## Run on the rented Ubuntu instance

Use a CUDA **devel** image with `nvcc` (CUDA 12.6 or newer), a compatible GCC
compiler with C++20 support, CMake 3.24 or later, Python 3, and `nvidia-smi`.
Ubuntu 24.04 includes a suitable CMake version. Install any missing host tools
and run the tests from the repository root.

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake python3
nvcc --version
nvidia-smi
bash cuda/run_tests.sh --output-dir /tmp/aster-a100-run-001 --sanitizer required
```

Pick a new output directory under an existing parent. If a run fails, the
directory keeps its partial results. The runner needs one GPU, and its Python
steps use only the standard library.

The runner builds in Release mode for `sm_80` and checks the command-line options,
overwrite protection, and numerical comparisons. It runs hand-checked examples,
the full correctness suite (including input reloads and output reuse), and Compute
Sanitizer memcheck. It records 20 timing samples after warm-up.

The runner starts one GPU process at a time. Device `0` is the first visible GPU.
The results include a list of processes seen on the device so you can check for
other users. Those snapshots cover only the times when the list was collected.

The default, `--sanitizer auto`, runs memcheck if installed and records whether it
ran. Use `--sanitizer required` to fail the run if the tool is missing.
`--sanitizer off` skips memcheck and records that choice. Memcheck changes
execution times, so use the separate benchmark samples for performance comparisons.

To time a different workload, set its dimensions and launch settings explicitly.

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

The default mode is `--mode smoke`. Omit `--output` to print JSON on standard
output. The program creates the output file before initializing the GPU and
refuses to overwrite an existing file. A failed run may leave an empty file.
A successful JSON report contains `"passed": true`. You can run `--help` and
check invalid command-line arguments without a GPU.

One hand-checked case expects P&L of 0.089, costs of 0.011, and turnover of 1.1.
Other cases cover shorts, one interval, zero holdings or costs, empty batches,
partial blocks, 1–256 intervals, and sums that cancel exactly or nearly cancel
in gross gain or final P&L. Random prices range from 0.5 to 2, holdings from
−1 to 1, and cost rates from 0 to 0.01. After each timed repeat the program
compares every output with the CPU result.

One GPU thread processes each path in time order. The CUDA build disables fused
multiply-add contraction and fast arithmetic. The host build uses
`-ffp-contract=off -fno-fast-math`. Keep these flags when reproducing the tests.

## Read the evidence

`benchmark.json` stores the individual timings. Each field covers a different
part of one batch calculation.

| Field | What the timer covers |
| --- | --- |
| `kernel_event_ms` | Kernel only, CUDA events in the same stream, read after completion |
| `completed_host_ms` | Host-to-device copies, kernel, all three device-to-host copies, stream synchronization |
| `native_cpu_prepared_ms` | Single-threaded CPU calculation and output writes, timed after allocation and input checks |
| `cold.runtime_setup_ms` | First runtime/device initialization and hardware queries |
| `cold.device_allocation_ms` | Reusable device buffers, stream, and events |
| `cold.first_completed_host_ms` | First small smoke call, including transfers and first kernel launch |

The benchmark reuses device buffers and events. It uses pageable host memory, so
host timing includes any temporary copies the driver needs for transfers. Input
creation, allocation, JSON formatting, correctness checks, and checksums happen
outside the warmed samples. The cold fields time individual setup stages. The
initial smoke call uses a different workload size from the warm benchmark.

The output directory includes device, driver, runtime, and compiler details,
a process list, build commands and logs, source and binary SHA-256 hashes, and
the sanitizer result. The JSON also records the seed, array dimensions, launch
settings, checksums, errors, and raw times. The input hash uses FNV-1a over the
host bytes for the input arrays and cost rate. Keep this directory with any
published results.

## Resident accounting sessions

This benchmark holds prices and positions fixed while changing the transaction
cost rate. For timings that include simulation, neural network training, search,
and an independent check of the winner, see the full workflow.
[CUDA workflow guide](../docs/CUDA_WORKFLOW.md) · [Full workflow measurements](../docs/evidence/cuda-launch-2026-11-04/README.md)

Run `--mode pipeline` and set `--operations` to the number of rates (K), from 1
to 256. The default is 8. For one rate, the program uses 0.01. For multiple rates,
they span 0 to 0.01. Each rate uses the same float64 kernel. Every method returns
P&L, costs, and turnover for every path, with its output arrays ordered by
`[operation][path]`. The CPU runs the full calculation for each rate without
caching gain or turnover between calls.

```bash
/tmp/aster-cuda-build/aster_cuda --mode pipeline --paths 65536 --intervals 12 \
  --operations 32 --block-size 128 --warmups 5 --repeats 15 \
  --output /tmp/aster-pipeline-001.json
```

| Pipeline field | What the timer covers |
| --- | --- |
| `copy_per_call_ms` | K operations, each uploading both inputs, running one kernel, downloading three output arrays, and synchronizing |
| `resident_session_ms` | One input upload, K kernels writing distinct retained slots, three bulk downloads covering every operation, and one synchronization |
| `resident_preloaded_ms` | K kernels, all output downloads, and synchronization, with inputs already on the device |
| `native_cpu_sequence_ms` | K prepared native calls and all output writes on one CPU thread |

Compare `resident_session_ms` with `copy_per_call_ms` to see how long the full
set of calculations takes with each transfer method. The resident session uploads
the inputs once, copies results back in larger batches, and waits for the GPU
once. Both methods use a host clock and run without CUDA timing events. The
single-kernel benchmark reports CUDA event times in `kernel_event_ms`.

The program allocates input arrays, output arrays, device slots, and sample
storage before warm-up. Each copy-per-call download writes directly to its slice
of the host output array. Host vectors use pageable memory, so host times include
any staging by the driver. The cold fields time device setup, host preparation,
device allocation, and the first single-operation call at the chosen size.
They cover only those stages of startup.

The pipeline report uses JSON schema version 2. Smoke, correctness, and
single-kernel benchmark reports use version 1. The pipeline report records the
exact `workload.rates`, input hash, device and build details, and raw
`pipeline_samples`. Each sample records its reproducibly shuffled `method_order`.
After timing each repeat, the program compares every result using the same
tolerances as the other modes. It saves numeric sums for each method and an
FNV-1a hash of the final checked outputs. It hashes P&L first, then costs, then
turnover, preserving operation and path order. A CPU sample shorter than one
clock tick can appear as zero. Each completed GPU sample must be positive.

The `transfers` field counts successful `cudaMemcpyAsync` requests. For N paths,
T intervals, and K rates, the input size is `I = N * (2*T + 1) * 8` bytes and
the output size is `O = K * N * 3 * 8` bytes. H2D means host to device, and D2H
means device to host.

| Method | H2D bytes / calls | D2H bytes / calls |
| --- | --- | --- |
| `copy_per_call` | `K*I` / `2*K` | `O` / `3*K` |
| `resident_session` | `I` / `2` | `O` / `3` |
| `resident_preloaded` | `0` / `0` inside timing | `O` / `3` |

The counts give the bytes requested by the code. Measuring traffic on the PCIe
bus would require separate instrumentation. Upload inputs again whenever they
change, even if their dimensions stay the same. Uploading invalidates previous
results. The session checks rates, output slots, and dimensions before calling
CUDA. It also rejects attempts to launch before upload, read incomplete outputs,
or read outputs a second time without running another kernel.

The default correctness suite tests single and multiple rates, hand-checked
outputs, short positions, cancellation, partial blocks, empty batches, new
dimensions, and reloading different inputs with the same dimensions.
`--sanitizer required` also runs these cases under Compute Sanitizer. The
benchmark covers reuse of fixed inputs. Python calls, training, and
hyperparameter search fall outside the timed work.

At 65,536 paths and 32 operations, uploading prices and holdings once reduced
session time from 26.41 ms to 4.42 ms. The median speedup across paired runs was
5.97×. With one operation the two methods took about the same time. The report
includes the 36-process protocol, results, checks, and commands.
[A100 session report](../docs/evidence/cuda-resident-2026-11-04/README.md)

## Host-only checks

These tests run with a C++20 compiler and standard Python.

```bash
python3 -m unittest discover -s cuda/tests -v
mkdir -p cuda/.build-local
c++ -std=c++20 -Icuda/include cuda/tests/test_comparison.cpp \
  -o cuda/.build-local/test_comparison
cuda/.build-local/test_comparison
bash -n cuda/run_tests.sh
```

The comparison test corrupts each output and checks that the comparator catches
errors, including nonfinite values and values outside the near-zero tolerance.
The shell tests cover command-line arguments and check that existing output
files survive a second run.

[NVIDIA guide to asynchronous execution and events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) · [Compute Sanitizer error exit codes](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
