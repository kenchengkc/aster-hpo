# CUDA Workflow Implementation Plan

> Use executing-plans for continuous implementation in the existing worktree;
> independent CUDA implementation and benchmark collection can run concurrently.

**Goal:** Execute the actual hedging simulation and HPO objective on CUDA and measure search plus independent audits.
**Spec:** ../specs/2026-11-04-cuda-workflow-design.md
**Architecture:** Shared low-level accounting kernel, optional resident CuPy trainer,
serializable objective backend, fresh-process full-workflow benchmark.
**Tech stack:** Python/NumPy/CuPy, C++20/CUDA, existing Study/SQLite runtime.

## Constraints and review focus

- Preserve existing NumPy defaults, seeds, float64 math and dependency-free core.
- No per-epoch host array round trips or scalar synchronizations on CUDA.
- Explicit CUDA errors, correct device/current stream, shape/dtype checks.
- Same random inputs, gradient/parameter/metric equivalence; evaluation cannot affect training.
- Backend-aware provenance/resume; GPU host CPU seconds are not device compute cost.
- Compare real full-budget equal work before interpreting adaptive throughput.

## Task 1: Resident training and backend integration

Files: cuda/include/accounting_kernel.cuh, cuda/hedge_accounting.cu,
benchmarks/hedging_cuda.py, benchmarks/hedging.py, examples/hedging.py,
benchmarks/protocol.py, benchmarks/study.py, tests/test_hedging_cuda.py,
tests/test_hedging.py, tests/test_study.py.

- [ ] Write and observe failures for backend validation/dispatch, shared accounting and numerical GPU tests.
- [ ] Add shared kernel, resident CUDA simulation/gradients/Adam/evaluation with NumPy seed compatibility.
- [ ] Add backend selection to objective/example/frozen-study and backend provenance.
- [ ] Run local regressions and real A100 accounting/gradient/training/Study checks, including spawned evaluation and memcheck.
- [ ] Commit tested integration.

## Task 2: Full-workflow benchmark

Files: benchmarks/cuda_workflow.py, tests/test_cuda_workflow.py.

- [ ] Write failing tests for raw record integrity, failed/incomplete observations, timing scope, paired summaries and preserving output directories.
- [ ] Implement fresh campaign processes using real Study, winner lock, audit and SQLite journals; freeze the spec's protocol before timing.
- [ ] Record cold setup, search/audit/workflow/process durations, losses, source and GPU identities; use correct paired summaries.
- [ ] Execute all 30 campaigns on the current A100 host only after numerical checks pass.
- [ ] Commit collector and archive unedited raw results locally.

## Task 3: Review and report

- [ ] Run appropriate full local regression suite, lint/format and real GPU verification.
- [ ] Request one independent whole-change review, address substantive findings with regressions.
- [ ] Audit raw results against protocol; report tiny/scaled and CPU four-worker controls, independent audit quality and limitations.
- [ ] Archive checksummed evidence/plots/reproduction, update README/guides/roadmap/changelog, commit and retain branch.
