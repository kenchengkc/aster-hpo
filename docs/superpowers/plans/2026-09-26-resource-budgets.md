# Resource Budgets Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Add enforceable local run limits and cancellation with durable recovery.
**Architecture:** Session controls govern coordinator admission; owned spawn workers support
preemption; SQLite records attempt and session outcomes. Existing unlimited serial calls remain inline.
**Tech Stack:** Python 3.11+, standard-library multiprocessing, threading, SQLite.
**Spec:** ../specs/2026-09-26-resource-budgets-design.md

## Global Constraints

- Dependency-free core, stable seeds, one replication per candidate in flight.
- Keep the v0.2 evidence unchanged and retain the existing SQLite study manifest.
- Per-invocation allowances; never fabricate CPU usage or losses for killed attempts.
- Descriptive branch names and neutral project authorship; no publishing or merging.
- Continue inline as in the previous phase; use one fresh final reviewer.

## Review Focus

- Cancellation after accepted progress and before any dispatch must both be resumable.
- Attempt caps include retries but deliberate cancellation does not exhaust infrastructure retries.
- All IPC and shutdown waits must allow the coordinator to enforce deadlines.
- A completed session must not hide pending work or select a partial candidate as winner.
- Fatal serialization/worker errors must leave a recoverable committed journal and no live workers.

## Task 1: Session control and journal contracts

**Files:** src/aster_hpo/control.py, storage.py, core.py, __init__.py; tests/test_control.py.
**Interfaces:** RunBudget frozen dataclass, CancellationToken.cancel()/is_cancelled, optimize keyword arguments;
SQLiteStore.total_attempts(), event(kind,payload), stop_attempts(ids,status,error,snapshot).

- [x] Write validation, zero-dispatch pre-cancellation, serial max_attempts/resume and session-history tests.
- [x] Run tests/test_control.py; expect missing APIs.
- [x] Implement per-call admission controls and transactional cancelled/timed_out attempt statuses.
- [x] Exclude cancelled attempts from the infrastructure retry count, retain them in total dispatch counts.
- [x] Run the full suite and commit the tested control contracts.

## Task 2: Supervised workers and preemption

**Files:** src/aster_hpo/workers.py, runtime.py; tests/test_budgets.py.
**Interfaces:** Worker owns spawn process, isolated connection and parent exchange thread; run_parallel
uses controller checks for wall deadline/cancellation and per-dispatch timeout.

- [x] Write real subprocess tests for exact parallel admission, worker death/retry, timeouts,
  token cancellation, wall expiry, and absence of surviving managed workers.
- [x] Run new tests; expect unsupported supervised time controls.
- [x] Implement persistent workers, nonblocking coordinator mailboxes, bounded group cleanup,
  per-worker replacement, cancellation snapshots, and failed-trial timeout handling.
- [x] Preserve completed observations and committed-only interruption cleanup.
- [x] Run all tests and commit the executor integration.

## Task 3: Fault boundaries, documentation, and verification

**Files:** tests/test_budgets.py, examples/budgeted_search.py, README.md, docs/RUNTIME.md,
ROADMAP.md, CHANGELOG.md, pyproject.toml, src/aster_hpo/_version.py.

- [x] Add regressions for SIGTERM resistance, serialization errors, cancellation/resume at zero retries,
  and timeout followed by successful work with one worker. Verify each missing behavior before fixes.
- [x] Add a budgeted/resumable example and document stop semantics, latency bounds, backend changes,
  partial-winner behavior, attempt/CPU distinctions, and deferred descendant-process control.
- [x] Bump to 0.3.0; retain historical performance claims at their recorded revision.
- [x] Run tests, Ruff, package build/install, and the example. Inspect exported session records.
- [x] Request one fresh whole-phase review; fix important findings with regression tests.
- [x] Commit verified changes and retain the branch/worktree for review.

## Completion record

All three tasks are complete on `feat/resource-budgets`.

- Full suite: 119 passed; Ruff lint/format and whitespace checks passed.
- Wheel and source archive built; installed the wheel in an isolated environment
  without optional dependencies and ran the budget/resume example successfully.
- The final review found one deadline-ordering defect. A regression reproduced an
  on-time result being discarded during another worker's cleanup. Parent mailbox
  timestamps now preserve on-time responses while rejecting genuinely late ones;
  both regression variants pass. Persistent callable state is documented.
- Recorded demonstration exports and checksums are in
  `docs/evidence/runtime-2026-09-26/`; their execution revision remains `6eec3d2`.
- The archived v0.2 financial report was regenerated and matched its saved summary.
  Its timing evidence remains specific to its recorded implementation.
- Keep this branch and managed worktree for review; publishing and merging remain
  outside this phase's scope.
