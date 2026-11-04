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

## Implemented in v0.3

- [x] Per-invocation wall deadlines and exact evaluation-attempt admission caps.
- [x] Caller cancellation and per-dispatch evaluation timeouts with owned workers.
- [x] Session limits/outcomes in the journal and resume after deliberate stopping.
- [x] Bounded worker cleanup, kill escalation, and isolated worker replacement.

## Runtime extensions

The [native performance foundation](docs/NATIVE.md) adds optional C++20 accounting,
strict Python bindings, CPU measurement tools, and a standalone CUDA test package.
An [A100 validation run](docs/evidence/a100-2026-11-04/README.md) passed correctness
and memory checks; initial timings expose transfer overhead. Later builds add
[resident accounting sessions](docs/evidence/cuda-resident-2026-11-04/README.md) to
amortize input copies and readback/synchronization. Full GPU training,
kernel-setting optimization studies, Rust replay, and Go orchestration remain future
work; see the [native design](docs/superpowers/specs/2026-11-04-native-performance-design.md).

- [ ] Measured CPU-time budgets and broader system-level resource accounting.
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
