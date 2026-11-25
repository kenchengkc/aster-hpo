# Aster

Aster is a Python library for finding good model settings when simulation results vary from run to run. For example, a hedging model can show a small error because it learned a good strategy or because it drew a favorable set of prices. Aster can compare model settings on the same test paths, stop spending runs on settings that perform poorly, and check the winner on fresh simulations.

[Project website](https://aster-hpo.vercel.app/) · [CPU study](docs/RESULTS.md) · [CUDA study](docs/evidence/cuda-launch-2026-11-04/README.md)

## Run a search

The core library requires Python 3.11 or later and has no runtime dependencies. The hedging example uses NumPy and runs on a laptop CPU. Run these commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[benchmark]'
python -m examples.hedging --workers 2 --configs 4 --output results/hedging-pilot
```

The example tries four sets of model settings. It runs two searches on the same simulated market. One gives each candidate nine runs. The other can drop candidates after three. Both searches then evaluate their chosen model settings on fresh simulations. Open `results/hedging-pilot/summary.json` to compare the search times, number of runs, and mean squared hedging error in the final checks. Use a new output directory each time you run the example.

## Use your own simulation

Write a function that takes a configuration and a replication (one run of the simulation) and returns a finite loss. Lower values are better. Here the function measures how far a setting is from a randomly drawn target. Candidates with the same replication index see the same target.

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

`ReplicationRace` compares candidates after 3, 9, and 27 runs in this example. It uses the difference in their average losses and the uncertainty in that difference to decide where to spend more runs. It can discard the best candidate when losses are close. `FullBudget` evaluates every candidate equally. A third option, `ASHA`, ranks candidates at each checkpoint and stops those that rank poorly. [Scheduler details](docs/BASELINES.md)

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

**CPU search.** In a simulated options market on an Apple M5, racing averaged 25.75 runs per search compared with 54 for a fixed budget, a **52.3% reduction**. Both methods used four workers. Across 20 random seeds, racing achieved a **1.43× median search speedup**, excluding the final audit. Both methods chose the same settings for all 20 seeds and had identical audit results. How often they would agree on other markets remains an open question. [CPU results and limitations](docs/RESULTS.md) · [Reproduction instructions](docs/BENCHMARKING.md)

**CPU and GPU search times.** A separate study on one NVIDIA A100 PCIe 40 GB host used three random seeds and 42 fresh processes across all comparisons. The table shows median seconds for a fixed-budget search and its audit. Each search tested four configurations four times each, then ran three independent checks on the winner. Both GPU columns used one worker.

| Simulated paths per training batch / evaluation | NumPy, 1 worker | NumPy, 4 workers | Original CUDA | Optimized CUDA |
| --- | ---: | ---: | ---: | ---: |
| 128 / 2,048 | 1.032 s | 0.742 s | 4.625 s | 0.923 s |
| 4,096 / 32,768 | 16.720 s | 8.865 s | 5.264 s | 3.604 s |

Four CPU workers finished the smaller search and audit fastest. On the larger workload, optimized CUDA was **4.64× faster than one CPU worker** and **2.57× faster than four**. These speedups use the median of the comparisons for each seed. Relative to the original CUDA code, the optimized version achieved a 4.96× speedup on the smaller workload and 1.46× on the larger one.

The times include setup, data transfers, training, evaluation, scheduling, database writes, and the audit. Including Python startup and exit brings the small workload to 1.165 s on optimized CUDA and 1.124 s on one CPU worker. All seven financial measures agreed within the study's numerical tolerances. In a separate profile of one warmed-up small evaluation, the GPU executed 361 kernels, down from 10,985. [CUDA results, profiles, and reproduction instructions](docs/evidence/cuda-launch-2026-11-04/README.md)

**GPU accounting.** A separate test reused the same 65,536 price paths for 32 transaction-cost calculations. Uploading the inputs once cut the time from 26.41 ms to 4.42 ms compared with uploading them for each call. This measurement covers the accounting calculations and all result transfers. Timing started after the GPU and its buffers were ready. [Accounting benchmark](docs/evidence/cuda-resident-2026-11-04/README.md)

## Persistent workers

**The repeated-run performance study is pending.** `WorkerPool` is available for searches and audits that share worker processes. A GPU worker can keep its CUDA context and compiled kernels while each evaluation starts with fresh model weights and price paths. The planned study compares direct calls, starting a pool for each run, and reusing a pool. It will measure the first run, later runs, and total time including shutdown. [Worker pool usage](docs/WORKER_POOLS.md) · [Benchmark protocol](docs/PERSISTENT_BENCHMARK.md)

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
