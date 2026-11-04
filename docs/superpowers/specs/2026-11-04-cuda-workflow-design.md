# CUDA simulation and HPO workflow

The owner requests integrating the CUDA backend into the actual simulation and
optimization pipeline and benchmarking the complete workflow on the rented A100.
Continue in the existing isolated `feat/native-performance` worktree. No new paid
resources, instance changes, or publication are needed.

## Approach

Keep the NumPy reference and its explicit analytic backpropagation. Add a lazy,
optional CuPy implementation of float64 path simulation, neural forward/backward,
gradient clipping, Adam, and evaluation. Extract the existing accounting kernel
into a shared CUDA header consumed by both the standalone executable and CuPy
RawKernel. Launch against resident device pointers on CuPy's current stream.
A host-copy-only accounting replacement cannot keep training resident; a PyTorch
rewrite would change the existing explicit gradient architecture unnecessarily.
This is the implementation choice under the owner's instruction to proceed.

## Contract

`HedgingObjective(market, diagnostic_dir, backend='numpy')` remains serializable.
`backend='cuda'` loads CuPy only inside evaluation. Explicit CUDA requests fail
clearly when unavailable, never silently falling back. NumPy remains the default.
Expose backend selection in the runnable hedging example and frozen study command.
Backend identity belongs in protocols and resume metadata; CUDA source belongs in
fingerprints. Keep the core package dependency-free. GPU benchmarks run from a
source checkout with the appropriate optional CuPy wheel installed.

Use the same NumPy initialization and random stream as the reference. CUDA prepares
all epoch shocks on the host once and uploads them once, then forms prices on the
device. This bounded host RNG preparation is included in full objective timings;
it preserves identical stochastic inputs and is not claimed as GPU random-number
generation. Prices, holdings, parameters, gradients, Adam moments and accounting
outputs stay on device during training. Collect finite-state flags on device and
check before returning; do not introduce per-epoch host scalar conversions.
Evaluation uses the same scenario seeds and returns only final scalar metrics.
Preserve opening/rebalance/liquidation costs, sigmoid, clipping, regularization,
Adam constants, float64, no accounting fast math, and sign(0)=0.

Check native kernel/device arrays, streams, shape/dtype/contiguity, empty and partial
blocks; compare accounting, full gradients, trained parameters, and metrics to
NumPy. Verify independent evaluation seeds and no look-ahead. A real Study, SQLite
journal, winner lock, independent audit and spawned-worker execution must work.
One GPU owner is used for benchmarks; CPU multiworker results are a separate control.

## Frozen development benchmark

Before comparative timings, freeze source hashes, protocol, job order and environment.
Two workloads: unchanged Market defaults (64 epochs, batch128, evaluation2048,
12 steps), and batch4096/evaluation32768 with the other defaults unchanged.
Three paired optimizer seeds 31000..31002; four candidates from the existing pool;
FullBudget(4) and ReplicationRace((2,4)); three independent winner audits per run.
Run NumPy/one worker and CUDA/one worker for both schedulers; additionally NumPy/four
workers for FullBudget. Thirty fresh campaign processes, interleaved deterministically.
The benchmark is a development comparison, not a formal non-inferiority study.

Time full cold workflow from before backend initialization through search, locking,
audit, completed device work and result writes. Record search, audit, initialization,
whole process and workflow durations separately. CPU pool startup and GPU compilation/
context setup are included where they occur. Use one BLAS thread per CPU worker.
No objective warmup before cold workflow timing; record cache/environment scope.

Compare fixed-budget matched losses, winners, audit means and observation counts;
report adaptive runs separately because asynchronous ordering can change pruning.
Every replication and audit must succeed. Paired optimizer-seed speedups use ratios
within seeds, descriptive ranges, no p99 claim. Include original tiny workload even
if CUDA loses. Compare four-worker CPU FullBudget against CUDA too. Report quality
numerically with a predeclared development closeness check (rtol1e-6, atol1e-8 for
matched losses), without claiming a general financial quality guarantee. Archive raw
results, source/dependency/device identities, sanitizer checks and reproduction.
