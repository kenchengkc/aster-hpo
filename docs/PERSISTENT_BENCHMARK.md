# Repeated HPO benchmark package

Status: the rented instance was paused before the repeated-run performance study.
No persistent-worker speedup is reported. Before the pause, the working snapshot
passed 94 integration tests on the A100, with two CPU-only checks skipped, including
actual CUDA child reuse, objective rebinding, audits, cancellation and resume. Its
passing summary was captured in the session; the detailed remote log was not
copied locally. Final packaged GPU validation should be rerun on the next host.

## Run on a compatible CUDA development host

Extract the source bundle and enter its project directory, then:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,benchmark,cuda13]'
.venv/bin/python -m pytest tests/test_worker_pool.py tests/test_hedging.py \
  tests/test_hedging_cuda.py tests/test_cuda_fused.py tests/test_study.py \
  tests/test_persistent_workflow.py -q
.venv/bin/python -m benchmarks.persistent_workflow run \
  --output results/repeated-workflows
```

Keep the validation log and use a fresh output directory. The CLI freezes source
hashes, library/device identities, job order, and protocol before timing. A resumed
or replacement instance may have different CPU/GPU resources and libraries; run
all arms on that host instead of treating older host timings as paired controls.
The earlier A100 used CUDA 13.2 and CuPy 14.2.0. Recorded dependency details remain
in the existing evidence manifests.

## Fixed comparison

The package executes 48 fresh coordinator processes. Each runs three independent
search-plus-audit workflows, giving 144 workflows and 2,736 expected evaluations.
Each search has four candidates and four replications; each winner has three
independent audits. All workflows use 64 epochs and 12 trading intervals.

| Workload | Training paths | Evaluation paths |
| --- | ---: | ---: |
| Original | 128 | 2,048 |
| Larger | 4,096 | 32,768 |

For each workload, three sequence roots (41000, 42000, 43000) run eight variants:
direct NumPy/one worker and direct CUDA/one worker; fresh pools for NumPy/one worker,
CUDA/one worker and NumPy/four workers; reused pools for the same three settings.
All audits request one worker, including from a pool with capacity four.

The candidate pool is sampled once from each sequence root. Study seeds are root,
root+1 and root+2, with new storage paths and fresh objective bindings. This holds
the model-width mix fixed while requiring new training and audit simulations.
The already persistent direct GPU path and equally persistent CPU pools are
explicit controls.

## Timing and quality interpretation

The first workflow includes numerical imports and pool construction. Subsequent
workflow latency includes search, journal writes, winner locking, independent
audits, diagnostics and result writes. Fresh-pool latency also includes closing
that pool. Reused-pool final shutdown is reported separately and included in full
sequence and amortized cost. Parent process time includes interpreter startup and
exit. Compare all three scopes; later-run latency alone omits lifecycle costs.

There is no objective warmup before a sequence. Correctness checks and prior work
may populate the ordinary shared compiler/disk cache; fresh processes do not mean
empty JIT caches. First-versus-later times are descriptive. Lifecycle benefits are
compared at matching positions and sequence roots against fresh-pool and direct
controls. Each sequence contributes one median from its two later workflows;
three outer sequences are the timing units, rather than six independent warm runs.

The collector checks all seven matched financial metrics at `rtol=1e-6, atol=1e-8`,
accepted observation counts, fresh journals, worker PIDs and per-process nonces,
unique run leases, actual CPU/GPU execution, native thread settings, and bounded
cleanup. It emits direct reused-CPU1/reused-CPU4 versus reused-GPU1 comparisons.
Valid winner differences retain their own audit identities; missing fixed-budget
search evidence still fails validation. No global environment or opaque profiling
capture is needed.

Archive the output directory before deleting the next instance. Source snapshots
without `.git` have null Git metadata; map the recorded per-file hashes to the
packaged commit to establish provenance. See the [worker-pool guide](WORKER_POOLS.md)
and [design](superpowers/specs/2026-11-04-persistent-workers-design.md).
