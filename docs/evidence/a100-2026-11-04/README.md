# A100 hardware validation — November 2026

The tests ran at revision `880b83c` on an NVIDIA A100 PCIe 40 GB (compute
capability 8.0). The host ran Ubuntu 24.04 with CUDA 13.2.86, driver 615.71.10,
GCC 13.3.0, and CMake 3.28.3. The kernel used float64 arrays with one row per
path and 128 threads per block. The build disabled fast-math and fused multiply-add
contraction.

This run tested the accounting kernel and timed four batch sizes. The measurements
cover batch accounting only. Training, model quality, trading latency, and 99th
percentile latency fall outside this test.

## Validation

- All 4 smoke cases passed, with 21 scalar comparisons.
- The full suite passed **98 cases and 22,641 scalar comparisons** across P&L,
  costs, and turnover. Cases included opening and closing trades, rebalancing,
  short positions, empty batches, partial blocks, 1–256 intervals, and sums that
  cancel exactly or nearly cancel.
- Compute Sanitizer memcheck reported **0 errors and 0 leaked bytes** on that suite.
  Memcheck covered memory errors. Race detection was outside the scope of the run.
- The native accounting test passed on Linux with AddressSanitizer and
  UndefinedBehaviorSanitizer. The archive includes the compile commands, build
  cache, and list of linked libraries.
- All **35 Python extension tests passed** in a fresh Linux environment with Python
  3.12.3 and NumPy 2.4.1. GCC 13.3.0 compiled the extension.

The largest absolute difference across all checks was `1.3877787807814457e-17`.
The checks included comparisons against hand-written decimal values as well as
CPU/GPU comparisons. The largest error was about `1.4e-6` times the allowed
difference (`atol=1e-12, rtol=1e-10`).

## Observed batch timings

Times are medians in microseconds. Each workload used 12 intervals and five warm-up calls.

| Paths | Prepared native CPU | GPU kernel only | Completed GPU host call | Samples |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.040 | 10.240 | 52.677 | 30 |
| 256 | 2.516 | 10.240 | 56.400 | 30 |
| 2,048 | 19.969 | 10.976 | 74.068 | 30 |
| 65,536 | 645.388 | 15.456 | 803.303 | 20 |

CPU timing started after allocation and input checks. CUDA events in the same
stream timed the kernel. The host clock also included copies to and from pageable
host memory and the wait for the GPU to finish. The measurements reused device
buffers and recorded initial setup separately. With transfers included, GPU calls
took longer than the CPU calculation at all four sizes. This prompted a separate
test of reusing inputs on the GPU across calls.

All measurements came from one session on one host, with the CPU timed first.
CPU affinity, clock rates, power, temperature, and competing work could vary.
The GPU process list was empty before and after the run. Other processes could
have run between those checks. Timer overhead matters for the 40 ns single-path
CPU measurement.

## Records and reproduction

`raw-records.tar.gz` contains the JSON measurements, source and binary hashes,
compile commands, hardware details, sanitizer logs, Python test environment and
results, and smaller batch runs. `summary.json` lists the results and
`SHA256SUMS` contains file checksums.

Run these commands from revision `880b83c` on a host with the CUDA development tools.

```bash
bash cuda/run_tests.sh --output-dir /tmp/aster-a100-run-001 --sanitizer required
mkdir /tmp/aster-a100-crossover
for count in 1 256 2048; do
  /tmp/aster-a100-run-001/build/aster_cuda --mode benchmark \
    --paths "$count" --intervals 12 --block-size 128 --warmups 5 --repeats 30 \
    --output "/tmp/aster-a100-crossover/paths-${count}.json"
done
```

The guides describe the build and the CPU and GPU timing methods.
[CUDA guide](../../../cuda/README.md) · [Native interface and timings](../../NATIVE.md)
