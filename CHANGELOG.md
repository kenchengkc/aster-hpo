# Changelog

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
