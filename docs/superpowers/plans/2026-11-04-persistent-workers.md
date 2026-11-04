# Persistent workers Implementation Plan

> Execute continuously under the owner's instruction. Worker lifecycle and repeated
> benchmark are independent implementation tasks with explicit interfaces below.

**Goal:** Reuse supervised GPU/CPU processes across independent HPO runs and audits.
**Spec:** ../specs/2026-11-04-persistent-workers-design.md
**Architecture:** Owned spawn WorkerPool, exclusive per-run lease and token/fresh
objective binding, unchanged scheduler/journal/seeds, repeated-workflow collector.
**Tech stack:** Standard-library multiprocessing/threading, existing runtime, optional CuPy objective.

## Constraints and review focus

- No pool leaves all existing behavior intact; core stays dependency-free.
- Reject invalid/closed/concurrent/foreign-process use before storage writes.
- Nonce protects cross-run attempt-ID reuse; fresh objective binding and native thread limits.
- Retire dead idle workers; timeout/cancel/error never leaves active work or reused pipes.
- Pool capacity may exceed requested workers; active concurrency obeys study.workers.
- Equal CPU persistence and direct GPU controls; include first-use/close/amortized costs.

## Task 1: Reusable pool and public API

Files: src/aster_hpo/workers.py, optional new pool module, core.py, runtime.py,
__init__.py, tests/test_worker_pool.py.
Interface: WorkerPool(workers=1,threads_per_worker=1), Study.optimize(...,worker_pool=None),
audit(...,worker_pool=None). Healthy children persist between leases; close bounded.

- [x] Observe failing tests for public API and actual spawned PID/callable reuse.
- [x] Implement lazy pool, token-bound fresh objective binding, child thread limits,
  scheduler capacity, cancellation/timeouts/recovery/cleanup and metadata.
- [x] Test deterministic equivalence, rebinding, stale responses, ID reuse, attempt
  drain/resume, crash/timeout/cancellation, invalid use, thread control and clean exit.
- [x] Verify legacy tests and commit tested implementation.

## Task 2: Repeated HPO evidence

Files: benchmarks/persistent_workflow.py, tests/test_persistent_workflow.py,
root-owned GPU integration tests/usage example if needed.
Interface: existing HedgingObjective, FullBudget4, audit3; worker_pool keyword above.

- [x] Write failing collector tests for48jobs/three workflows, first/subsequent and
  whole-sequence boundaries, same-seed quality, incomplete/reused evidence and PIDs.
- [x] Implement fixed eight variants and safe worker-side identity diagnostics;
  keep GPU initialization inside real evaluating process and recorded timing.
- [x] Run real A100 pool-reuse/changed-objective/budget checks before performance collection.
- [ ] Freeze sources/settings/order and execute48 fresh sequence processes.

## Task 3: Review and publish

- [x] Full local regression/lint and actual A100 verification; one independent review.
- [ ] Audit source/PID/journal/metric integrity and paired sequence summaries.
- [x] Package source and benchmark runbook, update guides/README/changelog/roadmap,
  commit and retain worktree/branch. Instance paused by owner.
- [ ] Collect and publish performance report/plot/raw evidence after GPU availability returns.

## Implementation checkpoint — instance paused

- Local combined suite: 306 passed,42 optional GPU cases skipped; lint/format clean.
- Working snapshot A100 validation before pause:94 passed,twoCPU-onlychecks skipped,
  including actual GPU worker reuse/rebinding/audits and initialized-context cancel/restart.
  Passing summary captured in the session; detailed remote log was not downloaded.
  Final packaged GPU validation should be rerun when an instance is available.
- Independent review found observation-PID validation and equal-persistence summary
  gaps. Both were fixed after failing regressions;22collector tests now pass, with
  the complete local suite green. Core lifecycle review had no unresolved findings.
- 48-sequence/144-workflow benchmark is implemented and packaged but NOT executed.
  No persistent-worker performance claim is made. Continue from the
  [runbook](../../PERSISTENT_BENCHMARK.md) on a resumed or replacement host.
- Source and all earlier completed benchmark evidence are local. Retain branch/worktree;
  do not resume, rent, or delete GPU resources without the owner's instruction.
