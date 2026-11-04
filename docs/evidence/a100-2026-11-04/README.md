# A100 hardware validation — November 2026

Source revision `880b83c` was compiled and executed on an NVIDIA A100 PCIe 40GB,
compute capability 8.0, with CUDA 13.2.86, driver 615.71.10, GCC 13.3.0,
CMake 3.28.3, and Ubuntu 24.04. The build uses float64, path-major inputs, block
size 128, and disabled fast-math/FMA contraction.

This is development hardware validation and a small diagnostic timing sweep.
It does not establish whole-training speedup, trading latency, p99, or general
financial quality preservation. No financial model was trained on the GPU.

## Validation

- Smoke: 4 cases and 21 scalar comparisons passed.
- Correctness: **98 cases and 22,641 scalar comparisons passed**, covering all three
  outputs, opening/rebalance/liquidation, shorts, empty batches, partial blocks,
  1–256 intervals, and exact/near cancellation.
- Compute Sanitizer memcheck: the same suite passed, with **0 errors and 0 leaked
  bytes**. This does not establish a general race-free guarantee.
- Native Linux AddressSanitizer/UndefinedBehaviorSanitizer: accounting CTest passed.
  Saved compile commands/cache and linked libraries substantiate instrumentation.
- Fresh Linux Python 3.12.3/NumPy 2.4.1 environment: **35 companion tests passed**,
  with the C++ extension compiled using GCC 13.3.0.

The largest aggregate absolute difference was `1.3877787807814457e-17`. This includes
hand-written decimal literal comparisons, so it is not specifically CPU/GPU
disagreement. Maximum normalized error was about `1.4e-6` of the allowed
`atol=1e-12, rtol=1e-10` threshold; this is not relative numerical error.

## Observed batch timings

Medians in microseconds; 12 intervals and five warm-up calls per workload.

| Paths | Prepared native CPU | GPU kernel only | Completed GPU host call | Samples |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.040 | 10.240 | 52.677 | 30 |
| 256 | 2.516 | 10.240 | 56.400 | 30 |
| 2,048 | 19.969 | 10.976 | 74.068 | 30 |
| 65,536 | 645.388 | 15.456 | 803.303 | 20 |

Prepared CPU excludes allocation/validation. Kernel timing uses completed same-stream
CUDA events. Completed host timing includes pageable input/output copies and
synchronization; device buffers are reused. Cold stages are recorded separately.
Transfers make the completed GPU call slower than prepared CPU accounting at each
recorded size. This motivates keeping data on the device across operations; these
four sizes do not establish a crossover threshold.

There is one host/session and a fixed CPU-then-GPU order. Affinity, clocks, power,
thermals, and host contention were not controlled. GPU process lists were empty at
the before/after snapshots; that does not prove continuous exclusive use. Timer
overhead especially affects the 40ns single-path CPU observation.

## Records and reproduction

`raw-records.tar.gz` includes raw JSON, source/binary hashes, compile commands,
hardware inventory, sanitizer logs, Linux Python setup/results, and smaller-batch
runs. `summary.json` is the compact summary; `SHA256SUMS` verifies artifact bytes.
SSH connection details and keys are not included. Records were downloaded locally.

From source revision `880b83c` on a compatible CUDA development host:

```bash
bash cuda/run_tests.sh --output-dir /tmp/aster-a100-run-001 --sanitizer required
mkdir /tmp/aster-a100-crossover
for count in 1 256 2048; do
  /tmp/aster-a100-run-001/build/aster_cuda --mode benchmark \
    --paths "$count" --intervals 12 --block-size 128 --warmups 5 --repeats 30 \
    --output "/tmp/aster-a100-crossover/paths-${count}.json"
done
```

Timings will vary. See the [CUDA guide](../../../cuda/README.md) and
[native interface/measurement guide](../../NATIVE.md).
