# Reuse managed workers across studies

`WorkerPool` owns spawned processes that can serve multiple `Study.optimize` and
`audit` calls. With CUDA objectives, this keeps the child CUDA context, compiled
kernels, and allocator caches alive between requests. Model parameters, random
paths, and optimizer state are still created afresh by each hedging evaluation.
The core pool has no GPU or numerical dependencies.

The existing unbounded `workers=1` path already runs in the caller. Repeated calls
in one Python interpreter therefore already retain CUDA caches. A pool adds an
owned process boundary with managed cancellation and timeouts; it is not required
merely to keep the caller's CUDA context alive.

## Run several independent searches

From the source checkout with the optional CUDA dependencies installed:

```python
from pathlib import Path

from aster_hpo import FullBudget, Study, WorkerPool, audit
from benchmarks.hedging import HedgingObjective, Market, hedging_pool

market = Market(batch_paths=128, evaluation_paths=2048)
configs = hedging_pool(41000, 4)

if __name__ == "__main__":
    with WorkerPool(workers=1, threads_per_worker=1) as pool:
        for seed in (41000, 41001, 41002):
            folder = Path("results") / f"request-{seed}"
            objective = HedgingObjective(market, folder / "metrics", backend="cuda")
            result = Study(
                configs, scheduler=FullBudget(4), workers=1, seed=seed,
                storage=folder / "search.sqlite",
                objective_id="hedging-cuda-market-v1",
            ).optimize(objective, worker_pool=pool)
            if result.best_trial is None:
                raise RuntimeError("Search did not complete a candidate")
            checked = audit(
                objective, result.best_trial.config, replications=3, seed=seed,
                storage=folder / "audit.sqlite",
                objective_id="hedging-cuda-market-v1", worker_pool=pool,
            )
            print(seed, checked.mean)
```

Choose an `objective_id` representing the market, backend, and implementation.
Fresh storage paths create independent studies; reusing storage resumes committed
work instead. Explicit seeds and new paths matter when benchmarking repeated runs.
The spawn main guard is required for script entry points. Objectives must be
importable and picklable; live device arrays and initialized CUDA objects should
remain inside the evaluating process.

## Ownership and limits

A pool's worker capacity and threads per worker are fixed positive integers. A
study may use at most that capacity and must request the same native thread count.
For example, a four-worker CPU pool can run a four-worker search and a one-worker
audit. Active concurrency follows the study's worker count. Use one objective
worker for the single-GPU benchmark; capacity is not GPU device placement.

Each invocation leases the pool exclusively and supplies a fresh callable binding.
Its request/response token is distinct from SQLite attempt IDs. Changed diagnostic
directories and market settings therefore reach the correct invocation, and an old
response cannot become a new study's observation. Arbitrary module globals remain
process-local; objectives must manage their own state and randomness deliberately.

Normal completion and drained attempt limits retain healthy idle children.
Evaluation timeouts retire the affected child. Cancellation, wall limits, and
transport/coordinator errors reap owned workers; the pool can then start fresh
children for a later invocation. Retry and resume rules remain the same. Timed
allowances include dispatch, lazy process startup, and communication.

The context manager closes and reaps children on exit. Explicit `close()` in a
`finally` block is also supported and is idempotent. Closed, concurrently borrowed,
foreign-process, or incompatible pools are rejected. Do not close a borrowed pool
from another thread; cancel the active study through its `CancellationToken`.

## Benchmark repeated complete workflows

```bash
python -m benchmarks.persistent_workflow run --output results/repeated-workflows
```

The fixed comparison includes direct, fresh-pool, and reused-pool CPU/GPU arms,
plus four-worker CPU controls. Every workflow performs a new search, winner lock,
and independent audits with fresh journals and simulation seeds. The candidate
pool is fixed within a sequence so changing model widths cannot masquerade as
startup savings. The first run, later runs, complete sequence, shutdown, and parent
process durations are reported separately. No objective warmup precedes timing.

Compare the reused GPU against equally reused CPU and the already persistent
same-interpreter GPU control. Amortized sequence cost includes pool creation and
shutdown; later-run latency alone does not describe a short-lived application.
