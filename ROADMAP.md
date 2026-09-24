# Roadmap

v0.2 delivers the local runtime and first reproducible CPU research study. Remaining
items are research directions, not promised release dates or achieved guarantees.

## Implemented in v0.2

- [x] Dependency-free package, typed random spaces, fixed candidate pools.
- [x] Full-budget, stopping-ASHA, and matched-prefix replication racing.
- [x] SQLite acceptance transactions, resume, attempt identities, bounded retries.
- [x] Thread controls, immutable experiment manifests, worker timelines, explicit unknown costs.
- [x] Shared/independent scenario and candidate-order diagnostic ablations.
- [x] Additive, heteroscedastic, near-tie, and finite-variance heavy-tail diagnostics.
- [x] CPU neural hedging with transaction costs, delta reference, gradient/accounting checks.
- [x] Locked winners, disjoint audits, paired analysis across independent optimizer runs.
- [x] Four-method allocation study and separate fixed-workload scaling measurements.
- [x] Raw evidence, standalone figures, technical report, report regeneration.
- [x] Single-bracket ASHA decision compatibility test against pinned Ray 2.58.0.

## Runtime extensions

- [ ] Hard deadlines, cancellation, and enforced wall/compute budgets.
- [ ] Distributed execution and durable multi-machine coordination.
- [ ] Schema migrations and a public event-replay API.
- [ ] Multiple concurrent replications per candidate, with explicit decision ordering.
- [ ] System-level resource measurement beyond returned objective CPU time.

## Research extensions

- [ ] Full Ray Tune and Optuna TPE comparisons, with proposer/runtime differences separated.
- [ ] Prespecified time-to-target, equal-wall-budget, and equal-compute studies.
- [ ] Explicit finalist-selection budgets and formal sequential guarantees under stated assumptions.
- [ ] Runtime-correlated noise and synchronous-dispatch ablations.
- [ ] Larger candidate pools, harder ranking tasks, additional hardware, and longer workloads.
- [ ] Stochastic volatility and broader financial stress tests.
- [ ] Cost-aware allocation and validated fidelity policies.

Release gates: meaningful tests pass, installed-package examples work, and empirical
claims link to raw records with their protocol and limitations. See [results](docs/RESULTS.md).
