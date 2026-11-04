# CUDA neural hedging and hyperparameter search

On the current A100, the larger full-budget search-plus-audit workflow runs in
5.44s versus 18.41s for one NumPy worker and 8.65s for four. Median paired speedups
are 3.38× and 1.58× respectively. The original small workload favors CPU.
[Results, quality checks, raw evidence, and timing scope](evidence/cuda-workflow-2026-11-04/README.md).

The optional CUDA backend runs the existing neural-hedging simulation, explicit
gradients, Adam training, and policy evaluation with resident float64 GPU arrays.
It launches the same accounting kernel used by the standalone CUDA executable.
The default backend remains NumPy.

## Install and run

From a source checkout on a CUDA 13 development host:

```bash
python -m pip install -e '.[benchmark,cuda13]'
python -m examples.hedging --backend cuda --workers 1 \
  --output results/cuda-pilot --configs 4 --batch-paths 4096 \
  --evaluation-paths 32768
```

Use a fresh output directory. Explicit CUDA selection requires a working NVIDIA
device and CuPy; an unavailable backend raises an error. For CUDA 12, install the
matching CuPy wheel instead of the `cuda13` extra; see the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).
The measured configuration uses CUDA 13.2 and CuPy 14.2.0 on an A100 PCIe 40GB.

The objective is a serializable callable usable by the regular optimizer:

```python
from aster_hpo import FullBudget, Study
from benchmarks.hedging import HedgingObjective, Market, hedging_pool

objective = HedgingObjective(
    Market(batch_paths=4096, evaluation_paths=32768), backend="cuda"
)
result = Study(
    hedging_pool(31000, 4),
    scheduler=FullBudget(4),
    workers=1,
    seed=31000,
    storage="results/cuda-search.sqlite",
    objective_id="hedging-cuda-float64-v1",
).optimize(objective)
```

Use one objective worker for one GPU. The benchmark also measures the CPU with
four workers separately. Device selection follows `CUDA_VISIBLE_DEVICES`; GPU
state initializes lazily inside the evaluating process, including spawned workers.
Include backend identity in a persistent study's objective identifier.

The frozen financial study supports the same backend selection:

```bash
python -m benchmarks.study freeze --pilot results/cuda-pilot \
  --protocol results/cuda-protocol.json --backend cuda --workers 1
python -m benchmarks.study run --protocol results/cuda-protocol.json \
  --output results/cuda-study
```

The pilot must use the same backend. Protocol fingerprints include CUDA source
and compiler settings. GPU identity and library versions are checked on resume.
The original CPU evidence and its scope remain recorded in their own artifacts.

## What stays on the device

Training prepares the exact NumPy random shocks and initialization used by the
reference, then uploads each once per objective. Host random-number generation
and these transfers belong to the complete objective cost. Price simulation,
positions, accounting, gradients, clipping, and optimizer moments run on device.
Finite-state flags accumulate on device and are checked before training returns.
Evaluation transfers final scalar metrics to the host.

Prices and holdings change during training. This workload therefore differs from
the standalone fixed-input cost-rate sweep. Shared CUDA accounting preserves
opening, rebalancing, and terminal liquidation costs, with the original float64
arithmetic and explicit gradient convention at zero trades.

## Full-workflow benchmark

```bash
python -m benchmarks.cuda_workflow run --output results/cuda-workflow
```

The collector freezes two workloads and three optimizer seeds before launching
30 fresh campaign processes. Each campaign runs search with SQLite journaling,
locks its selected configuration, and runs independent winner audits. It measures
NumPy and CUDA with full-budget and replication-racing schedulers, plus a separate
four-worker CPU full-budget control. Initial setup, search, audit, transfers,
completed device work, and artifact writes are included in the workflow boundary;
process startup/import time is reported separately by the parent process.

Fresh processes use the normal shared CuPy disk compilation cache, which correctness
validation may already populate. This measures a new process and context, without
claiming first-ever installation or compilation latency. Host CPU seconds record
CPU consumption; they do not measure GPU compute cost. Compare matched full-budget
work first; adaptive runs can differ in allocation and must also report audit quality.
