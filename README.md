# Aster

**Parallel hyperparameter optimization for noisy simulations, with adaptive replication and recoverable experiments.**

[Results and evidence](docs/RESULTS.md) · [Reproduce the study](docs/BENCHMARKING.md) · [Runtime guarantees](docs/RUNTIME.md) · [Roadmap](ROADMAP.md)

[![CI](https://github.com/kenchengkc/aster-hpo/actions/workflows/ci.yml/badge.svg)](https://github.com/kenchengkc/aster-hpo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Aster allocates repeated evaluations across a frozen pool of configurations. It runs
simulations in parallel and stops allocating work to candidates whose matched outcomes
look worse than their peers. Independent training seeds capture initialization noise;
shared scenario seeds support paired comparisons of exogenous simulation noise.

In a **20-seed CPU neural-hedging study**, replication racing used **52.3% fewer
replications** and achieved a **1.43× median paired search speedup** versus full-budget
search with the same four workers. Both methods selected the same configuration in
20/20 runs, giving identical independent audit outcomes. This is observed agreement
on one synthetic market using the recorded v0.2 runtime; the zero-variance comparison does **not** establish a general
quality-preservation guarantee. [Protocol, raw evidence, limitations →](docs/RESULTS.md)

![Fixed-budget hedging quality and search time](docs/evidence/cpu-2026-09-24/quality_time.png)

## What is implemented

- Local process execution with a coordinator-owned SQLite journal, resume, bounded
  infrastructure retries, and duplicate/stale-result rejection.
- Per-invocation wall/attempt budgets, evaluation timeouts, and caller cancellation
  through owned, terminable workers. [Budget and resume semantics →](docs/RUNTIME.md#resource-limits-in-v03)
- `FullBudget`, single-bracket stopping `ASHA`, and heuristic `ReplicationRace`
  schedulers operating on replication counts.
- Matched-prefix comparisons reconsidered when peers finish; already running pruned
  work retains its outcome and cost.
- Stable per-replication seeds, locked winners, and independently seeded audits.
- Native thread controls, attempt/event histories, worker timestamps, and explicit
  accounting gaps for interrupted work.
- A NumPy neural-hedging workload with proportional transaction costs, verified
  cash flows and gradients, and independent training/evaluation paths.
- Repeated-seed comparisons, noise diagnostics, standalone figures, raw artifacts,
  and CI configured for Linux/macOS and Python 3.11–3.13.
- A dependency-free core; numerical, plotting, and reference-baseline packages are optional.

## Quick start

Requires Python 3.11+. Run these commands from a repository checkout:

```bash
git clone https://github.com/kenchengkc/aster-hpo.git
cd aster-hpo
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,benchmark]'
python examples/noisy_quadratic.py --workers 2
python -m examples.budgeted_search --output results/budget-demo
python -m examples.hedging --output results/hedging-pilot
```

The quadratic example is a functionality demonstration: its cheap objective does not
measure useful parallel speedup. The hedging example is a disjoint pilot; follow the
[benchmark protocol](docs/BENCHMARKING.md) to freeze and execute a comparison.
Research modules live in the checkout; the installable wheel contains the core library.

## Library usage

```python
import random

from aster_hpo import Float, ReplicationRace, Study, audit, sample_configs


def objective(config, replication):
    scenario = random.Random(replication.scenario_seed).gauss(0, 0.1)
    training = random.Random(replication.training_seed).gauss(0, 0.02)
    return (config["x"] - 0.25) ** 2 + scenario + training


if __name__ == "__main__":
    configs = sample_configs({"x": Float(-2, 2)}, count=16, seed=17)
    result = Study(
        configs,
        scheduler=ReplicationRace(checkpoints=(3, 9, 27)),
        workers=4,
        seed=17,
        storage="results/search.sqlite",
        objective_id="quadratic-with-training-noise-v1",
        max_retries=1,
        threads_per_worker=1,
    ).optimize(objective)

    if result.best_trial is not None:
        checked = audit(
            objective,
            result.best_trial.config,
            replications=100,
            seed=17,
            storage="results/audit.sqlite",
            objective_id="quadratic-with-training-noise-v1",
        )
        print(result.best_trial.config, checked.mean, checked.standard_error)
```

Objectives return one finite scalar loss; lower is better. Use a top-level callable
and a main guard for process execution. Running the same persistent study again
resumes committed observations. Change `objective_id` when the objective, data, or
simulation contract changes. See [recovery semantics](docs/RUNTIME.md).

## How allocation works

```mermaid
flowchart LR
    POOL[Frozen candidate pool] --> COORD[Coordinator and SQLite journal]
    COORD --> WORK[Process workers]
    WORK --> OBS[Accept replication]
    OBS --> POLICY[Check available matched prefixes]
    POLICY -->|continue| COORD
    POLICY -->|prune| STOP[Stop new dispatches]
    OBS -->|full cap| COMPLETE[Completed candidates]
    COMPLETE --> LOCK[Lock best mean-loss configuration]
    LOCK --> AUDIT[Independent audit]
```

`FullBudget` gives every candidate the same cap. `ASHA` stops candidates using prior
rung ranks. `ReplicationRace` uses paired mean differences and standard-error margins
at configurable checkpoints. Completed candidates can supply comparison prefixes;
pruned and failed peers cannot. Selection considers completed candidates at a common
cap. [ASHA reference semantics →](docs/BASELINES.md)

## Statistical and engineering limits

The racing boundary is a **heuristic**, not a confidence sequence or a repeated-testing
guarantee. Near-tie diagnostics include runs where it prunes the true optimum.
`audit()` provides descriptive statistics; the research runner adds paired analysis
across independent optimizer seeds, with explicit incomplete, insufficient, and
zero-variance outcomes.

Execution is local, with at most one replication in flight per candidate. There is
no cluster executor or measured CPU-time cap. Wall limits and cancellation preempt
managed workers; OS scheduling, coordinator operations, and cleanup add latency.
They do not control arbitrary subprocesses or external services created by objectives.
Recovery provides exactly-once acceptance, while interrupted objectives may execute
again. CPU cost of unreturned work remains unknown. Native thread controls do not
make numerical results bitwise portable across hardware, and asynchronous completion
can change pruning decisions.

The financial workload uses a small policy in a synthetic GBM market. Results do not
measure live trading performance, stochastic-volatility robustness, or superiority
to complete Ray Tune or Optuna workflows.

## Development and further work

```bash
python -m pytest -q
ruff check .
ruff format --check .
python -m build
# Optional compatibility test against pinned Ray:
python -m pip install -e '.[baseline]'
python -m pytest tests/test_asha.py -q
```

The [roadmap](ROADMAP.md) separates shipped functionality from remaining research.
[DESIGN.md](DESIGN.md) retains the broader target architecture. The existing
[project website](https://aster-hpo.vercel.app) presents an earlier synthetic smoke
snapshot; the repository report above is the current evidence.

Research foundations: [ASHA](https://arxiv.org/abs/1810.05934),
[common random numbers](https://arxiv.org/abs/1910.09259),
[Deep Hedging](https://arxiv.org/abs/1802.03042), and
[confidence sequences](https://arxiv.org/abs/1810.08240).
These works motivate the project; their algorithms and results are not claimed as original.

Maintained by Ken Cheng. Licensed under the [MIT License](LICENSE).
