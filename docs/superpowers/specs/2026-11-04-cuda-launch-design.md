# Reduce CUDA launch overhead

The owner requests profiling and reducing CUDA kernel launch overhead, particularly
on the original small workload, followed by another benchmark on the rented A100.
Continue under that authorization in the existing feat/native-performance worktree.

## Profile and approach

Nsight Systems2026.1.1 diagnostics at source76da272/f7d5669 profile one warmed,
representative width8 objective, 64 epochs, training128/evaluation2048 and
training4096/evaluation32768. The small objective executes10,985 kernels,36.5ms
summed GPU kernel duration within264.8ms profiled host time. Large executes11,126
kernels,78.4ms within280.0ms. Tracing overhead affects these durations; use fresh,
unprofiled full workflows for performance claims. Retain raw Nsight reports,
SQLite, CSV and exact diagnostic script.

Implement fused float64 CUDA kernels rather than graphs of thousands of existing
temporary operations. Graphs would need stable buffer lifetimes and could add
material capture cost to first-use workflows. A custom optimizer kernel directly
combines finite checks, norm, clipping and Adam without a persistent shape cache.
For small training batches (N<=512, width<=64), use four kernels per epoch: causal
forward, accounting/error, deterministic gradients/loss, and optimizer. For larger
batches/widths keep the existing CuPy/cuBLAS loss/gradient and fuse only optimizer.
This threshold is an implementation choice fixed before new workflow benchmarks;
report both paths and any performance regressions. Do not tune it on final results.

## Numerical and lifecycle contract

Preserve float64, random seeds/inputs, sigmoid, explicit gradients, transaction
cost conventions, regularization, clipping at norm1, Adam constants/bias corrections,
and final scalar-only transfers. Compile custom kernels with the existing strict
options. Preserve the eager implementation unchanged as a diagnostic reference.
Public backend `cuda` selects optimized training; add `cuda_eager` for the original
training path. Both use the same evaluation path and GPU device identity, with
implementation identity recorded explicitly. NumPy remains the default.

Fused gradients reduce deterministically without floating-point atomics. Every
parameter gradient must be ready before any parameter update; w2 stays unchanged
while computing hidden-layer derivatives. Use sign(0)=0, zero opening/terminal
holdings and chronological accounting. Include regularization before clipping.
Check loss and unclipped gradients each epoch, accumulate a device failure flag,
and inspect only after training; check final parameter finiteness too. Preserve
finite-gradient/squared-norm overflow behavior (infinite norm clips to zero).

Allocate workspaces per training invocation, reuse through epochs, and retain no
live array pointers in global caches. Kernel objects may cache per device. Respect
current CuPy stream/device and positive integer validated dimensions. Arbitrary
valid widths outside the small fusion bound retain the eager gradient fallback.
Add independent NumPy/eager/fused gradient, loss, optimizer-update, 1/8/64-epoch
parameter/metric, clipping, partial block, zero-cost/reg, finite failure, repeated
invocation, and nondefault-stream checks. Preserve all existing tests.

## Comparative protocol

Reuse the existing protocol: tiny128/eval2048 and scaled4096/eval32768;64epochs,
12steps;seeds31000..31002;4candidates;FullBudget4 and Race(2,4);3independent audits.
Keep NumPy1 and NumPy4full controls. Add contemporaneous `cuda_eager` campaigns for
both schedulers/shapes/seeds, bringing the protocol to42 fresh processes. Freeze
all sources/settings/order before timing. Compare optimized/eager within seed as
well as optimized/CPU; report workflow and parent process durations. All allocation,
compilation/context, transfers, search, journals, selection and audits stay inside
the existing declared timing boundaries. Shared ordinary compiler cache is retained
and disclosed. No warmed-only result substituted for complete workflow.

Before benchmarking run real A100 numerical tests and memcheck with documented
explicit API reporting and test pool cleanup. Profile optimized and eager on the
same representative inputs and derive counts/durations from raw Nsight data.
Profiles are diagnostic, not final timing evidence. Fixed-budget comparisons must
match every observation identity and all seven metrics at rtol1e-6,atol1e-8;
record winner/audit/allocation agreement rather than assuming it. Preserve prior
immutable evidence, leave the instance/branch/worktree intact, archive results
locally, and report limitations including first-use costs and small-workload outcome.
