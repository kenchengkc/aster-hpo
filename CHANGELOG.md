# Changelog

## Unreleased — Native companion 0.1.0

- Add resident CUDA accounting sessions with K retained outputs, reload/lifecycle
  checks, requested-copy counters, and bulk all-output readback.
- Benchmark equal-work GPU paths across 36 fresh processes. The 65,536-path/K32
  setting records a 5.97× median paired session speedup, including initial upload
  and all output downloads with reusable capacity. [Results and scope](docs/evidence/cuda-resident-2026-11-04/README.md).
- Extend correctness to 117 cases and verify resident execution with Compute
  Sanitizer, preserving original modes and historical evidence.

- Add optional C++20 batch hedge accounting and a streaming ledger, with finite-input,
  overflow, overlap, and closed-state checks and opening/liquidation transaction costs.
- Add an independently installable NumPy companion with strict array validation,
  caller-owned output buffers, and native build metadata.
- Add separate prepared/checked native, Python API, and complete-objective measurement
  tools, preserving timer-resolution zeros and unavailable derived rates.
- Prepare an A100 CUDA correctness/benchmark runner with reusable buffers, CPU-reference
  comparisons, completed device/host timings, and explicit Compute Sanitizer status.
  An A100 run passed 98 correctness cases and Compute Sanitizer memcheck; raw
  [validation and timing records](docs/evidence/a100-2026-11-04/README.md) are archived.
  Native Linux ASan/UBSan and 35 Linux companion API tests also passed.
- Configure native Release and Linux sanitizer CI; keep the core dependency-free.

## 0.3.0 — Resource budgets and cancellation

- Add `RunBudget` and thread-safe `CancellationToken` controls on `Study.optimize`.
- Enforce per-invocation attempt admission caps, wall deadlines, and per-dispatch
  evaluation timeouts; retain incomplete trials for resume after deliberate stopping.
- Replace process-pool execution with owned spawn workers, isolated communication
  channels, parent I/O threads, and bounded termination/kill cleanup.
- Retry only the affected worker's unfinished replication after worker death.
- Record session limits, backend, outcome, and attempts. Cancelled work remains in
  cost records but does not consume infrastructure retries. Timeouts fail a trial
  without fabricating an observation or loss.
- Preserve late results from pruned work after the final surviving trial completes.
- Keep unlimited one-worker local callables supported; timed/cancellable calls use
  a managed process and require a picklable objective.
- Add a budget/resume demonstration and fault tests for worker cleanup and recovery.
  The archived v0.2 performance study is unchanged and does not benchmark this executor.

## 0.2.0 — Recoverable runtime and reproducible research

- Reconsider eligible matched-prefix comparisons when late peers arrive; retain costs
  of already dispatched work after pruning.
- Add SQLite journals, compatible resume, bounded worker-pool retries, native thread
  controls, and worker/attempt telemetry.
- Make locked-winner audits resumable in a disjoint seed namespace.
- Add stopping-ASHA with a pinned Ray decision-compatibility test.
- Add noisy diagnostic suites and a NumPy neural-hedging workload with transaction costs.
- Add frozen protocols, repeated-seed paired analysis, independent audits, fixed-workload
  scaling, and figures regenerated from raw artifacts.
- Protect interruption cleanup from speculative coordinator state and preserve exhausted
  trials in the recovery queue. Record the thread-control dependency for resume checks.
- Qualify zero-variance paired comparisons as degenerate, without an inferential bound.
  This correction was applied after the archived study; its original analysis is retained.
- Include repository evidence for 52.3% fewer replications and a 1.43× median paired
  search speedup on one four-worker synthetic-market study, with observed winner agreement
  in 20/20 runs. See [scope and limitations](docs/RESULTS.md).

## 0.1.0 — Initial research prototype

- Added frozen random configuration pools and deterministic replication seeds.
- Added local process parallelism, full-budget search, and heuristic paired pruning.
- Added isolated audit evaluation and atomic JSON result export.
- Added a synthetic example, tests, CI, and contributor documentation.
