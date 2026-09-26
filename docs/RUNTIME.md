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

Lost workers are replaced individually up to the per-replication retry budget. Objective
exceptions and non-finite values fail the trial and are not retried. Serialization
errors abort the campaign. Exhausted retries abort rather than fabricate a loss.
Managed workers are terminated during interruption cleanup. Timed limits and caller
cancellation are available through the session controls described below.

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

`audit()` accepts the same storage, objective_id, retry, worker, and thread options.
Use a separate database for audit; its replication identities retain the `audit`
phase. An audit locks one configuration and therefore runs at most one replication
at a time under the current per-candidate concurrency rule.

Interruption cleanup persists the last committed database state, not speculative
in-memory decisions. A failure before or during acceptance leaves that replication
eligible for retry; an interruption after commit preserves its accepted observation.
Exhausted retry entries remain visible on resume and fail explicitly. Recovery does
not silently report a campaign with missing pending candidates as finished.

## Resource limits in v0.3

```python
from aster_hpo import RunBudget

partial = study.optimize(
    objective,
    budget=RunBudget(
        max_wall_seconds=60,
        max_attempts=100,
        evaluation_timeout_seconds=10,
    ),
)
print(partial.metadata['termination_reason'])
# A new invocation has a fresh allowance and reuses committed progress.
continued = study.optimize(objective, budget=RunBudget(max_wall_seconds=120))
```

The controls are optional and apply to **one invocation**. They do not modify the
immutable study identity. Each invocation appends a `session_start` event with its
limits and backend, and a `session_end` event with its outcome and work count. A
coordinator killed before cleanup may leave an unmatched start event, recording an
unclean end. Session IDs refer to the corresponding start event sequence number.

| Control | Behavior |
| --- | --- |
| `max_attempts` | Stop admitting new evaluations after this many durable dispatch attempts, including retries; drain admitted work. |
| `max_wall_seconds` | Stop dispatching and terminate active managed workers when the invocation's wall allowance expires. |
| `evaluation_timeout_seconds` | Terminate an overdue worker and fail its trial. Timing starts at dispatch and includes startup and IPC. Other trials continue. |
| `cancellation` | A caller-supplied `CancellationToken`; calling `cancel()` stops active workers and preserves resumable progress. |

An attempt cap alone does not limit the duration of an admitted evaluation. Combine
it with a wall limit when an objective might hang. These are dispatch-count limits,
not measured CPU-second budgets. Limits must be finite and positive; booleans and
fractional attempt counts are rejected.

```python
from aster_hpo import CancellationToken

token = CancellationToken()
# A UI handler or another thread can call token.cancel().
result = study.optimize(objective, cancellation=token)
```

A token is one-way: reuse after cancellation stops immediately; use a new token to
resume. Supplying a token or any timed limit moves even a one-worker run to a managed
spawn process. Use a module-level objective and a main guard. Untimed one-worker runs
with only an attempt cap retain support for local functions and closures.

### Stopping and resuming

`metadata['termination_reason']` is `completed`, `attempt_budget`, `wall_budget`, or
`cancelled`. Completion means no pending/running work remains; inspect trial statuses
for numerical failures, timeouts, and pruning. Errors and KeyboardInterrupt propagate
rather than returning a normal result, and their session end records `error`.

A deliberately cancelled attempt is recorded as `cancelled`, its unfinished trial is
returned to `pending`, and its next invocation reuses the same replication seeds.
Such attempts count toward dispatch usage and unknown cost, but not the infrastructure
retry allowance. Already pruned trials remain pruned. A timeout is recorded as
`timed_out` and fails the affected trial without fabricating a scalar loss; it is not
a retry or a resumable pending trial. A full study reset requires a new database.

Completed observations commit once. A partial result can have no winner: `best_trial`
only considers configurations that reached the scheduler's full replication cap.
The ordinary independent `audit()` helper remains unbounded; this version adds run
controls to `Study.optimize`, not partially sampled audit statistics.

### Timing, ownership, and cost

The coordinator checks cancellation and deadlines at most every 20ms while waiting
for workers. It uses parent I/O threads so a partial result transfer does not block
these checks. Each worker has its own channel; forced termination discards that channel.
Python documents the risks of terminating a process using shared pipes, queues, or
locks, and that terminating a process does not terminate its descendants.
[Python multiprocessing reference](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Process.terminate).

All managed workers receive a stop signal together. After a shared 200ms grace period,
remaining workers are killed and given up to one shared second to exit; I/O threads
have a shared 200ms cleanup allowance. Cleanup failure raises an error. These are
objective-preemption controls, not real-time OS guarantees: scheduling, interpreter
startup/serialization, filesystem/SQLite operations, and cleanup can exceed a wall
allowance. Aster controls its worker processes, not arbitrary descendant processes,
external programs, or remote services started by an objective. Forceful termination
can interrupt objective side effects; make objective outputs atomic and retry-safe.

`session_attempts` is the current invocation's durable dispatch count.
`session_wall_seconds` includes session setup and worker cleanup. The existing
`wall_seconds` and allocated-worker fields remain cumulative coordinator-session
measurements, excluding downtime and setup before coordinator construction.
Returned observations supply measured objective CPU time. Killed or unreturned work
has unknown CPU cost, reported by `unmeasured_attempts`; no zero-cost assumption is made.

Run `python -m examples.budgeted_search --output results/budget-demo` for an actual
CPU-work example that stops on a wall allowance, resumes the same SQLite study with
zero infrastructure retries, and exports both session results.
