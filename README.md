# Aster

Aster runs parallel hyperparameter searches for simulations in Python. A hedge can look good on one set of simulated prices and do poorly on another. Aster can test candidate hedges on the same price paths, drop settings with higher losses, and check the winner on a fresh sample.

[Project website](https://aster-hpo.vercel.app/) · [CPU study](docs/RESULTS.md) · [CUDA study](docs/evidence/cuda-launch-2026-11-04/README.md)

## Run a search

The core library requires Python 3.11 or later and has no runtime dependencies. The hedging example uses NumPy and runs on a laptop CPU. Run these commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[benchmark]'
python -m examples.hedging --workers 2 --configs 4 --output results/hedging-pilot
```

The example compares two ways to search the same four configurations. The fixed-budget search gives each configuration nine runs. The racing search can drop a configuration after three. Each search then tests its winner on fresh simulations. Open `results/hedging-pilot/summary.json` to compare elapsed time, the number of runs, and the winners’ mean squared hedging errors. Choose a new output directory for each run.

## Compare hedging strategies

Run `python -m examples.hedging_controls` to compare a neural hedge with fixed settings, a Black–Scholes delta hedge, and an unhedged position. Each starts with the call premium. Trades cost 0.10% of the value traded, including the final trade to close the hedge.

With seed 42, the delta hedge had the lowest mean squared hedging error on 2,048 paths held out from training. Its error was 0.000385, compared with 0.000528 for the neural hedge and 0.014849 without a hedge. The example also prints average trading cost and turnover. To try another sample, change `--seed`. [Example source](examples/hedging_controls.py)

## Use your own simulation

Write a function that takes a configuration and a replication (one simulation run) and returns a finite loss. Aster minimizes that loss. In this example, the loss is the squared distance from a random target. Every configuration sees the same target for a given replication index.

```python
import random

from aster_hpo import Float, ReplicationRace, Study, audit, sample_configs


def loss(config, replication):
    target = random.Random(replication.scenario_seed).gauss(0.25, 0.1)
    return (config["x"] - target) ** 2


if __name__ == "__main__":
    configs = sample_configs({"x": Float(-2, 2)}, count=16, seed=17)
    study = Study(
        configs,
        scheduler=ReplicationRace(checkpoints=(3, 9, 27)),
        workers=2,
        seed=17,
    )
    result = study.optimize(loss)
    if result.best_trial is not None:
        winner = result.best_trial.config
        checked = audit(loss, winner, replications=100, seed=17)
        print(winner, checked.mean, checked.standard_error)
```

`ReplicationRace` compares candidates after 3, 9, and 27 runs. It can stop a candidate early if its average loss is high enough relative to the uncertainty in the comparison. When losses are close, it can drop the best candidate by mistake. `FullBudget` gives every candidate the same number of runs. `ASHA` ranks candidates at each checkpoint and stops the lowest-ranked ones. [Scheduler details](docs/BASELINES.md)

The search selects the lowest average loss among candidates that reached the final checkpoint. `audit` evaluates the selected settings using a separate set of random seeds. For longer studies, set `storage` and `objective_id` to save results in SQLite and resume after an interruption. You can also limit elapsed time, cap attempts, set a timeout for individual runs, or cancel a search. [Recovery and time limits](docs/RUNTIME.md) · [Stop and resume example](examples/budgeted_search.py)

## C++ and CUDA support

**C++20 accounting.** The optional C++ library computes profit and loss, transaction costs, and turnover from prices and positions. It handles opening a position, rebalancing it, and closing it. Python callers can process a batch of paths, supply their own output arrays, or use `Ledger` to update a position one time step at a time. Install it with `python -m pip install ./native` on a system with a C++20 compiler and CMake 3.20 or later. [C++ and NumPy API](native/README.md)

**CUDA training and evaluation.** The CPU and GPU versions use the same random inputs and starting weights. The CUDA version uploads them once per evaluation and keeps prices, positions, gradients, and model updates on the GPU. For small training batches it groups operations into four kernel launches per training step. The CPU version uses NumPy by default. [CUDA implementation and setup](docs/CUDA_WORKFLOW.md)

These commands run the hedging example on one NVIDIA GPU with CUDA 13.

```bash
python -m pip install -e '.[benchmark,cuda13]'
python -m examples.hedging --backend cuda --workers 1 --configs 4 --output results/hedging-cuda
```

## Benchmark results

**CPU search.** On an Apple M5 with four workers, racing used **52.3% fewer runs** than a fixed-budget search in the simulated options market (25.75 on average versus 54). Across 20 random seeds, the median search speedup was **1.43×**, excluding the final audit. For every seed, both methods picked the same settings and got the same audit result. The study covered one market. [CPU results and limitations](docs/RESULTS.md) · [Reproduction instructions](docs/BENCHMARKING.md)

**CPU and GPU search times.** On a machine with one NVIDIA A100 PCIe 40 GB, the study ran 42 fresh processes across three random seeds. The table shows the median time for a fixed-budget search and its audit. Each search tested four configurations with four simulations apiece, then ran three independent checks on the winner. Both CUDA versions used one worker.

| Simulated paths per training batch / evaluation | NumPy, 1 worker | NumPy, 4 workers | Original CUDA | Optimized CUDA |
| --- | ---: | ---: | ---: | ---: |
| 128 / 2,048 | 1.032 s | 0.742 s | 4.625 s | 0.923 s |
| 4,096 / 32,768 | 16.720 s | 8.865 s | 5.264 s | 3.604 s |

To reproduce the table, run this command from the repository root with Python 3.11 or later. It reads the saved timings and uses only the standard library.

```bash
python -m benchmarks.workflow_report
```

Four CPU workers finished the smaller search and audit fastest. On the larger workload, optimized CUDA was **4.64× faster than one CPU worker** and **2.57× faster than four**. Compared with the original CUDA code, the speedups were 4.96× on the smaller workload and 1.46× on the larger one. Each speedup is the median of three time ratios, one for each seed.

The times include setup, data transfers, training, evaluation, scheduling, database writes, and the audit. Including Python startup and exit brings the small workload to 1.165 s on optimized CUDA and 1.124 s on one CPU worker. All seven financial measures agreed within the study's numerical tolerances. In a separate profile of one warmed-up small evaluation, the GPU executed 361 kernels, down from 10,985. [CUDA results, profiles, and reproduction instructions](docs/evidence/cuda-launch-2026-11-04/README.md)

**GPU accounting.** A separate test reused the same 65,536 price paths for 32 transaction-cost calculations. Uploading the inputs once cut the time from 26.41 ms to 4.42 ms compared with uploading them for each call. This measurement covers the accounting calculations and all result transfers. Timing started after the GPU and its buffers were ready. [Accounting benchmark](docs/evidence/cuda-resident-2026-11-04/README.md)

## Persistent workers

**The repeated-run performance study is pending.** Searches and audits can reuse the same processes through `WorkerPool`. A GPU worker keeps its CUDA context and compiled kernels between evaluations. Each evaluation still starts with fresh model weights and price paths. The study will compare direct calls, a new pool for each run, and one pool shared across runs. It will time the first run, later runs, and the whole series through shutdown. [Worker pool usage](docs/WORKER_POOLS.md) · [Benchmark protocol](docs/PERSISTENT_BENCHMARK.md)

## Development

```bash
python -m pip install -e '.[dev,benchmark]'
python -m pytest -q
ruff check .
ruff format --check .
python -m build
```

[Contributing](CONTRIBUTING.md) · [Roadmap](ROADMAP.md) · [Design and research references](DESIGN.md)

Maintained by Ken Cheng. [MIT License](LICENSE)
