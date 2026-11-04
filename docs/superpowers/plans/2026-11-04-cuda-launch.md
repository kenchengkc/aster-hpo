# CUDA launch-overhead Implementation Plan

> Execute continuously under the owner's request. Parallelize independent fused
> kernels and benchmark collector; one final independent review and data audit.

**Goal:** Reduce launched work in resident training and rebenchmark full search plus audits.
**Spec:** ../specs/2026-11-04-cuda-launch-design.md
**Architecture:** Per-call GPU workspaces, four small-batch training kernels, fused
optimizer with existing gradients for larger shapes, preserved eager backend.
**Tech stack:** Python/CuPy, CUDA C++, Nsight Systems, existing Study/SQLite collector.

## Constraints and review focus

- Preserve seeds, float64, loss/gradients/optimizer conventions; NumPy default.
- No per-epoch host data or scalar transfers, unsafe cached addresses, or floating atomics.
- Same current stream/device; complete gradients before updating parameters.
- Include all first-use allocation/compilation and objective/workflow work in comparisons.
- Keep tiny/scaled settings fixed, controls contemporaneous, quality/failures checked.

## Task 1: Diagnose and implement launch fusion

Files: benchmarks/hedging_cuda_fused.py, cuda/include/hedging_training.cuh,
benchmarks/hedging_cuda.py, tests/test_cuda_fused.py.

- [x] Run baseline local suite and real small/scaled Nsight traces before implementation.
- [ ] Observe failing kernel/module tests; implement fused forward, accounting/error,
  deterministic gradient/loss reductions, finite/norm/clipping/Adam update.
- [ ] Preserve train_policy_eager, route optimized train_policy through small/fallback paths.
- [ ] Run CPU host checks and A100 numerical/parity/stream/lifecycle checks; run memcheck.

## Task 2: Wire explicit comparison and collect evidence

Files: benchmarks/hedging.py, benchmarks/study.py, benchmarks/protocol.py,
benchmarks/financial_report.py, examples/hedging.py, benchmarks/cuda_workflow.py,
tests/test_hedging.py, tests/test_study.py, tests/test_cuda_workflow.py.

- [ ] Test before adding explicit cuda_eager selection/provenance and 42-job protocol.
- [ ] Add paired optimized/eager summaries without changing legacy 30-job behavior.
- [ ] Profile final optimized/eager representative objectives and preserve raw reports.
- [ ] Freeze verified sources and execute all42 campaigns, with unchanged work/quality gates.

## Task 3: Review and publish

- [ ] Full local regression suite/lint, real A100 checks, independent code review.
- [ ] Audit42jobs/sourceidentities/rawjournals/metrics/summary and profile counts.
- [ ] Archive checksummed evidence/report/plots/reproduction; update docs, commit, retain branch.
