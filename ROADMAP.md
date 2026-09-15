# Roadmap

The versions below describe milestones, not promised release dates. The project design
assumes roughly 10–12 weeks for the first complete research study.

## v0.1 — Contributor foundation

- [x] Installable Python package with no runtime dependencies
- [x] Typed random search spaces and a frozen candidate pool
- [x] Local asynchronous process execution
- [x] Full-budget baseline and experimental paired replication pruning
- [x] Isolated audit seed namespace and descriptive summary
- [x] JSON observations, timings, failures, and decision records
- [x] Runnable synthetic example, tests, CI, MIT license, contribution guidelines

## v0.2 — Reliable experiment runtime

- [ ] Coordinator-owned SQLite storage with explicit state transitions
- [ ] Resume interrupted experiments; deduplicate replication results
- [ ] Attempt identities, bounded infrastructure retries, and stale-result rejection
- [ ] Cancellation, deadlines, compute-budget enforcement, and complete cost accounting
- [ ] Numerical-library thread limits and machine/configuration manifests
- [ ] Scheduler event replay and explicit result-schema versioning policy

## v0.3 — Reproducible comparisons

- [ ] ASHA implementation and established-tool baselines
- [ ] Fixed-budget finalist selection and independent paired non-inferiority analysis
- [ ] Controlled noise, near-tie, heavy-tail, and runtime-correlation test workloads
- [ ] Ablations for shared scenarios, uncertainty, and asynchronous dispatch
- [ ] Repeated-seed experiment runner and standalone figures

## v0.4 — Financial research benchmark

- [ ] Small neural hedging policy under proportional transaction costs
- [ ] Simulator checks and a delta-hedge reference
- [ ] Hyperparameter search with fixed simulation and training specifications
- [ ] Independent training replications and out-of-search evaluation paths
- [ ] Hedging-error, turnover, cost, and instability diagnostics
- [ ] Technical report with raw results and one-command reproduction

## Later research

- [ ] Optuna proposer adapter and Ray executor
- [ ] Formal selection procedure under explicit, testable assumptions
- [ ] Stochastic-volatility workload
- [ ] Cost-aware allocation and carefully validated fidelity policies

Release gates: tests and CI pass; examples run from the installed package; documentation
matches implemented behavior; empirical claims link to reproducible results.
