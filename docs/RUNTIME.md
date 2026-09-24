# Recoverable local experiments

```python
study = Study(
    configs,
    scheduler=ReplicationRace((3, 9, 27)),
    workers=4,
    storage="results/hedging.sqlite",
    objective_id="hedging-v1",
    max_retries=1,
    threads_per_worker=1,
    seed=17,
)
result = study.optimize(objective)
```

Run the same study again to resume committed progress. A completed database returns
its observations without rerunning the objective. Configuration order, seeds,
scheduler settings, workers, thread limits, retry policy, and objective identity
must match the saved manifest. Use a new database when changing them.

The objective_id is the caller's versioned contract for the objective, including
external data and simulation settings. The runtime cannot infer whether an
arbitrary callable or external dataset changed. Change the identity when its
meaning changes.

The coordinator owns SQLite writes and an operating-system file lock. This is
local single-coordinator storage, not a distributed database or a network-filesystem
coordination protocol. Accepted observations have unique trial/replication
identities. Observation acceptance, trial state, queue state, and pruning
decisions commit together. Duplicate and stale attempt results are rejected.
Observations are stored in separate rows rather than repeatedly rewriting history.

After interruption, completed observations are reused. Uncommitted evaluations may
run again with the same seeds. This is exactly-once **acceptance**, not exactly-once
objective execution; objectives should tolerate repeated evaluation.

Broken worker pools are restarted up to the per-replication retry budget. Objective
exceptions and non-finite values fail the trial and are not retried. Serialization
errors abort the campaign. Exhausted retries abort rather than fabricate a loss.
A graceful interruption can wait for running process-pool tasks to exit; hard task
deadlines and forceful cancellation are not implemented.

Every exported result includes attempts, submission/acceptance events, and objective
start/finish timestamps and process IDs. Objective CPU time sums returned
observations. CPU time from killed or interrupted attempts is unknown:
unmeasured_attempts exposes that gap, and objective_cpu_accounting_complete is
false. Allocated worker-seconds include coordinator overhead and idle capacity;
they are not measured CPU consumption. Across resumed sessions, wall time counts
active sessions, excluding downtime.

Native thread environment variables are set before workers spawn and restored on
exit. When installed, threadpoolctl also limits already loaded native pools.
Without it, preloaded libraries in serial execution may ignore environment
changes. This capability is recorded in metadata. Because environment variables
are process-global, run independent coordinators in separate processes rather
than concurrent threads in one process.

The store uses schema version 2. Other schemas or incompatible manifests are
rejected; automatic migration is not provided.
