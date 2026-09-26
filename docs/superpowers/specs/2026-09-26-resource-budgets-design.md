# Resource budgets and cancellation

## Intent and scope

Continue the first remaining runtime milestone after v0.2: resource limits that make
long simulations manageable and support future equal-wall-budget research. Preserve
the dependency-free Python 3.11+ library, durable acceptance, stable seeds, and honest
cost records. Implement inline in the existing isolated checkout. Keep the completed
v0.2 evidence immutable; no remote publication or merge is part of this phase.

## Choices

A cooperative stop flag alone cannot stop a hung objective. ProcessPoolExecutor does
not offer public worker termination APIs on the supported Python 3.11 floor. Use
owned spawn processes with isolated pipes and parent I/O threads so IPC does not block
the coordinator's deadline polling. Keep the unbounded one-worker in-process path for
local callables; timed or cancellable one-worker runs require an importable objective.

## Public contract

`RunBudget(max_wall_seconds=None, max_attempts=None, evaluation_timeout_seconds=None)`
is a frozen value object. Supplied seconds must be finite and positive; max_attempts
must be a positive integer. `CancellationToken.cancel()` is thread-safe and idempotent.
`Study.optimize(objective, *, budget=None, cancellation=None)` preserves existing calls.

Limits apply to one invocation, including resumed invocations. A new invocation can
supply a different allowance without changing objective identity. Persist each session's
limits, backend, start/end, attempt count, and termination reason in the event journal.
The existing immutable study manifest and SQLite schema remain compatible.

An attempt cap limits persisted dispatches, including infrastructure retries; drain
already admitted evaluations. Wall/cancellation stops terminate managed workers, preserve
committed observations, and requeue active nonpruned trials. Cancelled attempts have
unknown CPU cost and do not consume the infrastructure retry allowance. Evaluation
timeouts are measured from dispatch (including worker startup/IPC); fail that trial
without inventing an observation or retrying it. Other workers continue.

Return termination_reason completed, wall_budget, attempt_budget, or cancelled. A completed
session means no pending/running candidates, not that every candidate succeeded. A
partial result chooses a winner only among trials that reached the scheduler's full cap.
Errors and KeyboardInterrupt still propagate, after terminating workers and persisting
only committed state. A pre-cancelled token starts no evaluation.

## Runtime boundaries

Spawn one persistent process per active worker, at most one job per candidate. Replace
only a lost worker and retry its unfinished replication under existing retry rules.
Transport each request/result through a worker-owned channel, with attempt identity.
On shutdown signal all workers, wait against one shared grace deadline, then kill and
reap remaining processes. Discard channels after forced termination. Parent I/O threads
must exit when channels close; no unbounded shutdown waits.

This is objective preemption, not a real-time OS guarantee. Deadline checking, process
startup, serialization, SQLite/filesystem calls, and cleanup add latency. Aster controls
its workers, not arbitrary descendant processes or external services started by objectives.
An attempt budget is a work-count cap, not a measured CPU-time cap. Preserve these limits
in documentation; formal CPU-budget accounting and distributed execution remain future work.

## Validation

Test exact admission caps with multiple workers, safe repeated resume at max_retries=0,
real wall-time termination, caller cancellation, pre-cancellation, per-evaluation timeout,
worker death/retry, process cleanup including SIGTERM-resistant objectives, serialization
failure, pending work/winner semantics, journal session identities, committed-state failure
boundaries, and unchanged unlimited serial behavior. Run the existing numerical/research
suite, fresh-package installation, a budget/resume example, and one independent final review.
