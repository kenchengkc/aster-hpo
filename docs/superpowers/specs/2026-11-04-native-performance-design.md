# Native performance foundation and GPU direction

Date: 2026-11-04
Status: approved for implementation on 2026-11-04.
Base: `065020f` (Python package 0.3.0).

## Intent and agreed direction

Develop Aster into a stronger quant developer portfolio through measured latency,
memory use, numerical correctness, and CPU/GPU performance engineering. Keep Python
for experiments, allocation, seeds, recovery, audits, and analysis. The project owner
accepted the proposed profiling and C++20 batch/streaming foundation and confirmed
access to an NVIDIA GPU, locally or remotely. The owner subsequently rented an
A100 PCIe 40GB and requested a dedicated SSH key; connection details stay outside
the repository. Local inspection found an Apple M5 with Clang, without a CUDA
toolkit. Native build dependencies are installed in the project environment.

This document specifies the CPU foundation. On approval, the owner also requested
preparation of the standalone CUDA test package described below. GPU neural training,
launch autotuning, and additional language consumers remain follow-on directions.

## Approaches and scope

1. **Python orchestration plus a small C++20 core — selected direction.** Reuse the
   existing simulation and accounting reference; expose batch and streaming APIs.
   This makes numerical equivalence, allocation control, and interface costs
   measurable with a limited implementation.
2. **Move training and the executor to native code immediately.** Potentially affects
   more runtime, but changes optimization trajectories, cancellation, persistence,
   and deployment together. Defer until profiles justify a specific extension.
3. **Start with a GPU backend or separate services.** NVIDIA access makes CUDA
   feasible, but the current small arrays may not amortize launch and transfer
   costs. Establish the CPU reference and measurements first.

The first release adds neither a new executor nor a distributed service. The
dependency-free Python core, scalar objective protocol, SQLite manifest, existing
financial evidence, and process cancellation behavior remain compatible.

## Development profile

A development probe ran the existing, unmodified `HedgingObjective` on the local
machine with one numerical thread, 256 epochs, 256 training paths, 2,048 evaluation
paths, and 12 intervals. For widths 4/8/16, three warmed identical-input runs gave
median wall times of approximately 79/89/129 milliseconds. Repeated losses matched
within each configuration.

One separate width-8 cProfile run attributed approximately 6.3% of cumulative time
to `hedge_pnl`, 38.7% to `_forward`, and 79.7% to `loss_and_gradient`. These entries
overlap; they must not be added. Profiling changes timing, and three repeats do
not establish a performance distribution. This is development guidance, not a
speedup claim. [Probe settings and records](../../evidence/native-probe-2026-11-04/README.md).

The result supports accounting as a tractable systems component, not as evidence
that it dominates training. After native integration, remeasure the complete
objective before choosing a larger numerical component to optimize.

## CPU components and interfaces

### Core and batch accounting

Add a standalone C++20 library under `native/`, built with CMake. It must be usable
without Python. The initial implementation is scalar, single-threaded, and fused;
compiler vectorization can be examined later against that explicit baseline.

For N paths and T trading intervals, prices have shape `(N, T + 1)` and holdings
have shape `(N, T)`. Preserve the existing accounting:

- Gross gain is the sum of holdings times the next price change.
- Turnover charges absolute changes in holdings at the current price, including
  opening from zero and terminal liquidation to zero.
- Costs are turnover times a finite nonnegative cost rate; P&L is gain minus costs.

Use float64, contiguous path-major storage, finite inputs, and a fixed chronological
accumulation order. Support long or short holdings. Reject invalid shapes, negative
or nonfinite costs, and nonfinite inputs; report nonfinite output as an error.
An empty path batch is valid; at least one interval is required. Check shape/size
arithmetic before indexing. Do not modify input arrays.

Provide a checked batch entry point using caller-owned input/output buffers. Produce
one P&L, cost, and turnover value per path. The validated arithmetic loop must not
allocate. Public validation and output-allocation costs have separate measurements.
Input and output buffers must not overlap; checked wrappers reject overlap.

### Streaming ledger

Expose a small native ledger with price, holding, gross gain, turnover, cost rate,
and closed state. Begin with zero holding at an initial price. Each
`update(price, target_holding)` first marks the previous holding to the new price,
then charges the trade into the target holding at that price. For a batch path,
call update at each interval's starting price and holding, including the first
price. `finish(terminal_price)` marks the final interval and liquidates to zero.

Return P&L, costs, and turnover. Disallow updates or another finish after closing;
reset explicitly for reuse. Validate an event before mutating state. The successful
native update path performs no heap allocation and has bounded state. Batch and
streaming outputs for the same path must agree within the numerical contract.
This measures local accounting update latency, not exchange or trading latency.

### Python extension and optional C interface

Use nanobind for an optional companion package, `aster-hpo-native`, importing as
`aster_hpo_native`. Its Python-visible batch operation accepts NumPy float64 CPU
arrays with exact shapes and contiguous storage. Reject incompatible arrays rather
than silently converting/copying. Explicitly retain input owners during native work,
release the GIL only while avoiding Python API calls, and return NumPy-owned outputs.

Provide both an allocating convenience operation and an operation accepting
preallocated output arrays. Benchmark the latter's checked wrapper as well as the
standalone core. Python ledger calls allocate Python objects when returning state;
the allocation-free claim applies only to the native update, not its Python wrapper.

Keep the existing top-level Hatch build and core imports intact. The companion
package uses its own scikit-build-core/CMake configuration. A fresh environment
must still install and use the core without a compiler or native dependencies.
Explicit native selection fails clearly if unavailable; it never silently falls
back while labeling output native. Include backend/version in any new numerical
comparison manifest and use a new objective identity when changing implementations.

A thin C ABI is a possible later interoperability layer: caller-owned buffers,
explicit lengths, status codes, and exceptions contained within the C++ boundary.
It is not required to complete this first CPU milestone. No Rust or Go consumer
is introduced without a concrete use case.

## Correctness contract

The existing NumPy function is the mathematical reference. Feed exactly the same
saved input arrays to both implementations; matching seeds across languages is
not an equivalence test. Start with float64 without fast-math. Proposed comparison
tolerances are absolute `1e-12` and relative `1e-10`. The initial verified test
domain is prices in `[0.5, 2]`, holdings in `[-1, 1]`, cost rates in `[0, 0.01]`,
and 1 through 256 intervals. Include random and deliberately cancelling gain
sequences, with exact-zero and near-zero final P&L; the same absolute tolerance
still applies in those cases. Verify representative saved financial inputs too.
These bounds define the initial accuracy study, not additional public API input
restrictions. Stress cases outside this domain require a higher-precision scalar
oracle and separately reported errors before extending an accuracy claim.

Disable contraction into fused multiply-add for the first equivalence build;
record compiler settings. Floating-point reduction order may prevent bitwise
equality. If straightforward accumulation fails the declared accuracy contract,
use compensated accumulation rather than widening tolerances after the results.

Cover hand-calculated gain/cost/turnover, zero holdings/costs, one interval, opening,
rebalancing, liquidation, short positions, empty batches, reset/closed behavior,
randomized differential fixtures, invalid arrays, ownership/lifetimes, overlap,
and nonfinite inputs/results. Include native tests, Python differential tests,
installed-package checks, and AddressSanitizer/UndefinedBehaviorSanitizer runs.
There are no native threads in this milestone; add race testing when they exist.

Do not replace the training implementation as part of this milestone. A parity
result for accounting does not establish identical training trajectories, pruning
decisions, or general quality preservation.

## Measurements and acceptance

Preserve three independent boundaries:

| Boundary | Included work | Report |
| --- | --- | --- |
| Standalone native | Prepared inputs, core arithmetic, output writes | Time/path, throughput, allocations; streaming per-update samples |
| Python API | Validation, bindings, required allocation, native work | Completed call time and batch-size crossover |
| Complete objective | Training/evaluation, RNG, Python/native integration | Unprofiled wall/CPU time with unchanged financial settings |

Compare the existing NumPy implementation, straightforward C++, and each subsequent
optimization on identical work. Measure one path and the existing training/evaluation
sizes, plus a larger fixed workload. Report cold startup separately from warmed runs.
Include validation in API timing, consume outputs, and measure timer overhead.

Use individual completed events for latency quantiles: p50/p95/p99, sample counts,
quantile method, and variation across independent sessions. Batched average timing
does not establish per-event p99. Label sequential replay as service time. If a
later queued replay is added, preserve an external arrival schedule and include
backlog in arrival-to-completion latency. Do not drop scheduling outliers after
seeing the result.

Record source/binary hashes, compiler/version, build flags, floating-point settings,
architecture, CPU, thread pools, workload hashes, process counts, warmup policy,
and controllable host load/power settings. Hash native sources and build inputs,
not only the existing flat Python source inventory. Profile/instrumented builds
are distinct from uninstrumented release timing. Linux x86-64 validation can follow
local arm64 development; compare backends within the same machine.

The milestone succeeds with a correct installable native component, standalone
batch/streaming benchmarks, and reproducible measurements, including neutral or
negative outcomes. A speedup is not an acceptance condition. A whole-objective
claim requires a separate measured comparison at that boundary.

## Standalone NVIDIA CUDA package and follow-on integration

The first CUDA deliverable is a standalone correctness/measurement executable,
not a GPU training backend. Its path-major float64 accounting uses reusable device
buffers, the same CPU-reference inputs/outputs, completed CUDA-event/host timings,
exclusive evidence directories, and Compute Sanitizer checks on the A100. Remote
validation completed on the owner's A100; [records](../../evidence/a100-2026-11-04/README.md)
preserve correctness, sanitizer outcomes, and timing limits. No additional instance
was provisioned.

Start with device-resident batch hedge accounting, reusing explicit NumPy/C++ inputs
and all three outputs. Reuse device buffers and one GPU-owning process. Do not
multiply CUDA contexts by mapping the current CPU worker count directly onto one
GPU. Add more complex GPU simulation or policy computation only after profiling.

Sweep small and large batches to find the CPU/GPU crossover. The current training
sizes are included even if CPU wins. Compare existing path-major storage with
time-major storage that lets neighboring GPU threads read neighboring values.
Account for transpose/upload costs unless the producing stage already uses the
required layout. First use float64; GPU capability and speed depend on the actual
device. Treat float32 as a distinct accuracy experiment, not a hidden optimization.

Report device execution with CUDA events in the same stream after completion,
completed host-call wall time including transfers, cold initialization, and any
whole-objective time separately. An asynchronous enqueue duration is not completed
GPU work. No CUDA result can be validated on the local Apple GPU.

Use Aster itself for a later launch-configuration tuning experiment: vary block
size and paths per thread, while fixing arrays, path count, time steps, precision,
and accounting mathematics. Reject numerically incorrect configurations before
timing; use serial GPU ownership, randomized measurement order, independent finalist
remeasurement, and held-out input shapes. Keep this experiment separate from tuning
financial model hyperparameters. Do not claim that common seeds remove system
timing noise.

Before GPU implementation, collect NVIDIA model, compute capability, VRAM,
driver/toolkit, host compiler, remote access method, and sharing conditions.
Access to a GPU does not authorize cloud spending or remote deployment. Validate
CUDA memory/synchronization behavior with Compute Sanitizer on the target machine.

## Roles for additional languages

- **C++20:** initial CPU batch/streaming core and optional CUDA host code.
- **CUDA C++:** later NVIDIA batch kernels and measured launch tuning.
- **C:** small interoperability boundary when another-language consumer needs it.
- **Rust:** possible later bounded replay/parser or safe wrapper around that C ABI.
- **Go:** possible remote-job orchestration when multi-machine execution is added.

Rust replay and Go orchestration are alternatives for concrete future needs, not
additional numerical rewrites. Avoid adding independent runtimes solely to increase
the repository's language count.

## References

- [nanobind array ownership and constraints](https://nanobind.readthedocs.io/en/latest/ndarray.html)
- [CUDA asynchronous execution and event timing](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [CUDA memory and context guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [Clang AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html)

## Review and handoff

The owner approved this written design and authorized the CPU foundation and CUDA
test package. Rust, Go, the C ABI, GPU training, and launch autotuning retain their
separate future scopes. Builder-level choices such as internal file names,
test framework, benchmark harness, and packaging release mechanics may be settled
during planning without changing the numerical or measurement contract.
