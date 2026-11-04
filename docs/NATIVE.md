# Native CPU and CUDA measurements

Aster has an optional C++20 accounting library, Python companion package, and a
standalone CUDA test package. The dependency-free Python optimization core remains
separate. The first native component calculates hedge P&L, transaction costs, and
turnover, including opening and liquidation, on fixed input arrays.

The [CUDA workflow guide](CUDA_WORKFLOW.md) integrates the same kernel with resident
simulation, neural training, evaluation, and regular HPO search. Its
[30-campaign A100 comparison](evidence/cuda-workflow-2026-11-04/README.md) measures
the complete search-plus-audit workflow, including setup and transfers.

## Install and check the CPU foundation

From a repository checkout with Python 3.11+, a C++20 compiler, and CMake:

```bash
python -m pip install -e '.[dev,benchmark]'
python -m pip install ./native
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j
ctest --test-dir build/native --output-on-failure
ASTER_ACCOUNTING_BENCH="$PWD/build/native/accounting_bench" \
  python -m pytest tests native/tests cuda/tests -q
```

The [companion guide](../native/README.md) documents the allocating and preallocated
NumPy operations, streaming ledger, strict inputs, buffer ownership, and errors.
Read-only inputs are accepted; conversion/copying of incompatible inputs is
rejected. Outputs must not overlap inputs or each other. Arrays must remain stable
while the native function releases the Python interpreter lock.

## Measure the boundaries

```bash
python -m benchmarks.native_performance results/native-comparison \
  --standalone build/native/accounting_bench --objective
```

The destination must not already exist. The report retains exact input arrays,
workload hashes, numerical errors, raw timings, and source/build identity.
`--numpy-only` explicitly runs a reference-only probe when the companion is absent.
`--objective` compares the complete serial hedging objective using unchanged market,
training, and seed settings; it temporarily selects the accounting implementation
inside this benchmark process and restores it afterward.

| Boundary | Included work |
| --- | --- |
| Native prepared batch | Arithmetic and caller-owned output writes; validated inputs supplied by benchmark |
| Native checked batch | Input/shape/overlap checks and arithmetic |
| Python allocating API | Validation, bindings, output allocation, and arithmetic |
| Python preallocated API | Validation, bindings, and arithmetic; output allocation excluded |
| Native streaming update | One local accounting state transition, with individual event samples |
| Complete objective | Training, evaluation, RNG, and selected accounting implementation |

Fresh subprocess records measure cold API startup separately from warmed calls.
The `--sessions` option means measurement rounds in one process; repeat the whole
command into new output directories for independent process sessions. Warm method
order alternates between forward and reverse order. Native streaming samples retain zero durations when updates
fall below timer resolution; those samples are not replaced or discarded. Derived
rates with zero elapsed time are unavailable, not infinite throughput. Do not
interpret very small sample sets as reliable p99 estimates, or subtract timer
overhead from quantiles to manufacture operation latency.

The reference is existing vectorized NumPy, not an intentionally slow scalar
baseline. Accuracy uses identical arrays, float64, no fast-math, and disabled
multiply-add contraction. Batch tests use atol `1e-12`, rtol `1e-10` across the
declared financial test domain and include near-zero cancellation. A separate
Decimal fixture checks an outside-domain case; it does not extend the accuracy
claim to every finite input. Kernel agreement alone does not establish unchanged
training trajectories or independent financial quality.

## Prepare the rented A100

The [CUDA guide](../cuda/README.md) specifies the CUDA development image and tools.
The package builds one float64 path-major kernel for `sm_80`, compares all three
outputs against the CPU library, checks error/collision behavior, runs Compute
Sanitizer, and records repeated completed timings and machine/build metadata.

```bash
bash cuda/run_tests.sh --output-dir /tmp/aster-a100-run-001 --sanitizer required
```

One process owns the GPU. Kernel CUDA-event time, completed host calls including
transfers, reusable-buffer allocation, and cold initialization are separate fields.
Output correctness is checked outside timing. A GPU result requires actual
execution on NVIDIA hardware; host-only tests do not establish CUDA validation.
Keep the evidence directory and download it before removing the rented instance.

The [recorded A100 validation](evidence/a100-2026-11-04/README.md) passed 98 CUDA
correctness cases, Compute Sanitizer memcheck with zero errors/leaks, native Linux
ASan/UBSan, and 35 Linux Python companion tests. Its four-size timing sweep reports
kernel and completed-host times separately; transfers dominate these initial GPU
calls. The records are development observations, not whole-objective speedup claims.

The standalone CUDA executable measures accounting correctness/performance. The
separate optional CuPy backend provides GPU neural training through the shared
kernel. Current CPU financial timing evidence remains specific to its original
recorded implementation.

### Resident accounting sessions

`--mode pipeline --operations K` retains inputs across K cost-rate operations and
returns all three arrays for every operation. It compares copy-per-call execution,
an upload-once session with bulk readback, and a separately labeled preloaded-input
path. Context/capacity setup is outside these completed-session timings.

```bash
python3 cuda/benchmark_pipeline.py --binary build/cuda/aster_cuda \
  --output results/resident-sessions --paths 1 256 2048 65536 \
  --operations 1 8 32 --sessions 3 --repeats 15 --warmups 5
```

The [resident run](evidence/cuda-resident-2026-11-04/README.md) reduced the
65,536-path/32-operation session from 26.41ms to 4.42ms, with a 5.97× median paired
process speedup. Initial upload and every result download are included. K=1 is
approximately unchanged. The workload measures accounting data reuse; inputs that
change require a new upload.

## Build and verification scope

Build the native package with `python -m build native --outdir dist/native`.
The core wheel and companion wheel install independently; core-only environments
do not need NumPy or native build tools. Sanitizer checks use a separate build:

```bash
cmake -S native -B build/native-sanitizers -DCMAKE_BUILD_TYPE=Debug \
  -DASTER_ENABLE_SANITIZERS=ON -DASTER_BUILD_BENCHMARKS=OFF
cmake --build build/native-sanitizers -j
ctest --test-dir build/native-sanitizers --output-on-failure
```

Sanitizer builds are correctness checks, not timing evidence. Native CI includes
Linux/macOS Release builds and a Linux address/undefined sanitizer job; CUDA runs
are performed explicitly on GPU hardware. See the
[approved design](superpowers/specs/2026-11-04-native-performance-design.md) for
the full contract and deferred GPU/Rust/Go work.
