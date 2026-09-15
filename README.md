# Aster

**Parallel hyperparameter search for noisy simulations.**

[![CI](https://github.com/kenchengkc/aster-hpo/actions/workflows/ci.yml/badge.svg)](https://github.com/kenchengkc/aster-hpo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Aster explores how to spend less computation finding good configurations for stochastic
models. It runs independent evaluations in parallel and can stop configurations whose
paired results look clearly worse than their peers.

**Status: early research prototype.** Local parallel execution, random configuration
sampling, full-budget search, heuristic replication pruning, and an isolated audit helper
work today. The financial benchmark and comparative performance study are planned.
There are no established speedup or quality-preservation claims yet.

## Why build this?

One noisy evaluation can make a poor configuration look good—or make a good one look
bad. Evaluating every configuration many times is expensive. Aster's research question
is whether allocating additional replications to uncertain candidates can reduce time
and compute at a fixed result-quality target.

The planned flagship example tunes a neural hedging model under transaction costs.
The current synthetic example lets contributors exercise the scheduler without financial
data, GPUs, or runtime dependencies.

This library reduces the work in an optimization campaign; it does not accelerate an
individual simulation kernel. Existing systems already support parallel HPO. The intended
contribution is careful replication allocation and a reproducible evaluation of its tradeoffs.

## Install and run

Requires Python 3.11 or newer. Install from this repository; a PyPI release is not available.

```bash
git clone https://github.com/kenchengkc/aster-hpo.git
cd aster-hpo
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/noisy_quadratic.py --workers 2
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. Windows support has
not yet been verified in CI.

The example compares a fixed replication budget with adaptive pruning on the same
configuration pool. It writes raw observations, seeds, pruning decisions, audit observations,
and a summary to `results/smoke/`. This inexpensive objective is a functional smoke test:
process startup dominates its timing, so it is unsuitable for a speedup claim.

## Use your own objective

Save this as a Python file. Parallel objectives must be importable top-level functions,
and the entry point must use the `__main__` guard.

```python
import random

from aster_hpo import Float, ReplicationRace, Study, audit, sample_configs


def objective(config, replication):
    shock = random.Random(replication.scenario_seed).gauss(0, 0.1)
    return (config["x"] - 0.25) ** 2 + shock


if __name__ == "__main__":
    configs = sample_configs({"x": Float(-2, 2)}, count=16, seed=17)
    result = Study(
        configs,
        scheduler=ReplicationRace(checkpoints=(3, 9, 27)),
        workers=4,
        seed=17,
    ).optimize(objective)
    result.save("results/search.json")
    if result.best_trial is not None:
        check = audit(objective, result.best_trial.config, replications=100, seed=17)
        print(result.best_trial.config, check.mean, check.standard_error)
```

Return one finite scalar loss per replication; lower is better. Use
`replication.training_seed` for configuration-specific training randomness and
`replication.scenario_seed` for shared exogenous scenarios. Merely supplying a common
seed does not guarantee aligned randomness: the objective must consume the same
scenario stream consistently across configurations. Set `shared_scenarios=False` to
give each configuration independent scenario streams.

`FullBudget(replications=27)` evaluates all configurations to the same cap. `ReplicationRace`
compares candidates at replication checkpoints using matched replication indices. It
prunes when mean disadvantage minus `z` times its standard error exceeds `tolerance`.
Only surviving configurations that reach the cap are eligible for `best_trial`.

## Statistical and runtime boundaries

- **Pruning is heuristic.** Repeated inspection and adaptive peer selection mean `z=2`
  is not an overall 95% guarantee. Heavy-tailed losses can make standard errors unreliable.
- **Audit is separate from search.** Its seed namespace is disjoint. Choose its sample
  count in advance and lock the configuration first. The helper returns a descriptive
  mean and standard error, not a non-inferiority test. Reusing an audit to select winners
  invalidates its role as independent evidence.
- **Replications retain the same simulation specification.** Changing numerical resolution
  or training budget is a separate, currently unsupported fidelity policy.
- **Asynchronous outcomes can differ.** Objective seeds are deterministic, but completion
  order can change pruning. One worker provides a deterministic debug path for deterministic
  objectives.
- **Failures are visible.** Exceptions and non-finite losses fail the entire trial; earlier
  observations remain in the result. Failed trials are excluded from winner selection.
  Serialization errors and broken worker pools abort the run.
- **The runtime is deliberately small.** No deadlines, retries, resumable checkpoints,
  distributed executor, or automatic numerical-library thread limits yet. Set thread limits
  in your environment before importing numerical libraries to avoid oversubscription.

Study timing includes process startup and shutdown. Reported objective CPU seconds sum
worker CPU measurements around objectives; they exclude coordinator overhead, startup,
and any child processes launched inside an objective. They are not total allocated CPU-hours.
Search timings exclude the separate audit stage. JSON result files are final exports, not
resumable checkpoints.

## Research and development

Read the [design](DESIGN.md), [roadmap](ROADMAP.md), and
[benchmark protocol](docs/BENCHMARKING.md). The design describes the intended system;
the README and API describe what is implemented today.

```bash
pytest
ruff check .
ruff format --check .
python -m build
```

CI runs tests and the parallel example on Linux and macOS with Python 3.11–3.13.
The package has no runtime dependencies. Development tools are optional dependencies.

Contributions are welcome, especially reproducible failure cases, scheduler diagnostics,
and benchmark improvements. See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[community guidelines](CODE_OF_CONDUCT.md). Maintained by Ken Cheng. Licensed under [MIT](LICENSE).

## Research foundations

- [A System for Massively Parallel Hyperparameter Tuning](https://arxiv.org/abs/1810.05934): ASHA, a planned baseline.
- [Bayesian Optimization Allowing for Common Random Numbers](https://arxiv.org/abs/1910.09259): scenario reuse and correlated comparisons.
- [Time-uniform, nonparametric, nonasymptotic confidence sequences](https://arxiv.org/abs/1810.08240): a possible future statistical extension.
- [Deep Hedging](https://arxiv.org/abs/1802.03042): motivation for the planned financial workload.

These are prior work, not algorithms or results claimed as original to this repository.
