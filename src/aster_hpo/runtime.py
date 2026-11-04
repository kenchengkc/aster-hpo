"""Local coordinator, process execution, recovery, and thread controls."""

from __future__ import annotations

import os
import platform
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict

from ._version import __version__
from .control import SessionControl
from .core import Observation, Replication, ReplicationRace, StudyResult, Trial, _evaluate
from .schedulers import ASHA, ASHAPolicy, RacePolicy
from .storage import SQLiteStore
from .workers import WorkerPool, run_parallel

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextmanager
def thread_limits(count):
    """Set inherited worker environment, and loaded native pools when available.

    Environment variables are process-global: do not run concurrent studies in
    separate threads of one process. Independent coordinator processes are safe.
    """
    previous = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    limiter = None
    try:
        for name in THREAD_VARIABLES:
            os.environ[name] = str(count)
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            control = "environment only; preloaded native pools may ignore changes"
        else:
            limiter = threadpool_limits(limits=count)
            control = "environment and threadpoolctl"
        yield control
    finally:
        if limiter is not None:
            limiter.restore_original_limits()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class Coordinator:
    def __init__(self, study, store):
        self.study, self.store = study, store
        self.start = time.perf_counter()
        saved = store.load()
        self.prior_wall = saved["wall_seconds"] if saved else 0
        if saved:
            self.trials = [
                Trial(
                    t["id"],
                    t["config"],
                    t["status"],
                    [
                        Observation(**{**o, "replication": Replication(**o["replication"])})
                        for o in t["observations"]
                    ],
                )
                for t in saved["trials"]
            ]
            self.ready = deque(saved["ready"])
            for trial in self.trials:
                if trial.status == "running":
                    trial.status = "pending"
                if trial.status == "pending" and trial.id not in self.ready:
                    self.ready.append(trial.id)
            self.decisions = saved["decisions"]
        else:
            self.trials = [Trial(i, dict(config)) for i, config in enumerate(study.configs)]
            self.ready = deque(range(len(self.trials)))
            self.decisions = []
        if isinstance(study.scheduler, ReplicationRace):
            self.policy = RacePolicy(study.scheduler, self.decisions)
        elif isinstance(study.scheduler, ASHA):
            self.policy = ASHAPolicy(study.scheduler, self.decisions)
        else:
            self.policy = None
        store.recover(self.snapshot())

    @property
    def has_work(self):
        return any(t.status in {"pending", "running"} for t in self.trials)

    def snapshot(self):
        # Observations live in normalized rows, not the coordinator snapshot.
        return {
            "trials": [{"id": t.id, "config": t.config, "status": t.status} for t in self.trials],
            "ready": list(self.ready),
            "decisions": self.decisions,
            "wall_seconds": self.prior_wall + time.perf_counter() - self.start,
        }

    def dispatch(self):
        while self.ready:
            trial = self.trials[self.ready[0]]
            if trial.status != "pending":
                self.ready.popleft()
                continue
            replication = Replication.create(
                self.study.seed,
                trial.config,
                len(trial.observations),
                self.study._phase,
                self.study.shared_scenarios,
            )
            if self.store.attempt_count(trial.id, replication.index) > self.study.max_retries:
                raise RuntimeError(
                    f"Infrastructure retry budget exhausted for trial {trial.id}, "
                    f"replication {replication.index}"
                )
            self.ready.popleft()
            trial.status = "running"
            attempt = self.store.begin_attempt(trial.id, asdict(replication), self.snapshot())
            return trial, replication, attempt
        return None

    def accept(self, job, observation):
        trial, replication, attempt = job
        if observation.replication != replication or replication.index != len(trial.observations):
            raise RuntimeError("Out-of-order or mismatched replication")
        trial.observations.append(observation)
        if trial.status != "pruned":
            if observation.error is not None:
                trial.status = "failed"
            elif len(trial.losses) == self.study.scheduler.maximum:
                trial.status = "complete"
            else:
                trial.status = "pending"
                self.ready.append(trial.id)
        if self.policy is not None:
            for decision in self.policy.decisions(self.trials):
                self.decisions.append(decision)
                if decision["action"] == "prune":
                    self.trials[decision["trial_id"]].status = "pruned"
        if not self.store.accept(attempt, asdict(observation), self.snapshot()):
            raise RuntimeError("Stale or duplicate observation rejected")

    def retry(self, jobs, error):
        for trial, _, _ in jobs:
            if trial.status == "running":
                trial.status = "pending"
                self.ready.append(trial.id)
        for _, _, attempt in jobs:
            self.store.fail_attempt(attempt, "infrastructure_failed", str(error), self.snapshot())

    def cancel(self, jobs, reason):
        for trial, _, _ in jobs:
            if trial.status == "running":
                trial.status = "pending"
                if trial.id not in self.ready:
                    self.ready.append(trial.id)
        self.store.stop_attempts([job[2] for job in jobs], "cancelled", reason, self.snapshot())

    def timeout(self, job):
        trial, _, attempt = job
        if trial.status != "pruned":
            trial.status = "failed"
        self.store.stop_attempts(
            [attempt], "timed_out", "evaluation dispatch timeout", self.snapshot()
        )

    def interrupt(self):
        # In-memory changes may be speculative if dispatch/accept/retry raised
        # before their SQLite transaction committed. Cleanup must only preserve
        # committed state, including when an interrupt arrived just after commit.
        committed = self.store.load()
        committed["wall_seconds"] = self.snapshot()["wall_seconds"]
        attempts, _ = self.store.journal()
        for attempt in attempts:
            if attempt["status"] == "running":
                self.store.fail_attempt(
                    attempt["id"], "interrupted", "coordinator stopped", committed
                )


def run(study, objective, *, budget=None, cancellation=None, worker_pool=None):
    if worker_pool is None:
        return _run(study, objective, budget=budget, cancellation=cancellation)
    if not isinstance(worker_pool, WorkerPool):
        raise TypeError("worker_pool must be a WorkerPool or None")
    # Validate/acquire before opening SQLite, so invalid uses create no artifacts.
    with worker_pool._lease(study) as run_id:
        return _run(
            study,
            objective,
            budget=budget,
            cancellation=cancellation,
            worker_pool=worker_pool,
            run_id=run_id,
        )


def _run(study, objective, *, budget=None, cancellation=None, worker_pool=None, run_id=None):
    limits = SessionControl(budget, cancellation)
    manifest = {
        "schema_version": 2,
        "phase": study._phase,
        "configs": study.configs,
        "seed": study.seed,
        "scheduler": type(study.scheduler).__name__,
        "scheduler_parameters": asdict(study.scheduler),
        "workers": study.workers,
        "shared_scenarios": study.shared_scenarios,
        "objective_id": study.objective_id,
        "max_retries": study.max_retries,
        "threads_per_worker": study.threads_per_worker,
    }
    with (
        SQLiteStore(study.storage, manifest) as store,
        thread_limits(study.threads_per_worker) as control,
    ):
        coordinator = Coordinator(study, store)
        initial_attempts = store.total_attempts()
        if worker_pool is not None:
            backend = "persistent_processes"
        else:
            backend = (
                "in_process"
                if study.workers == 1 and not limits.needs_process
                else "supervised_processes"
            )
        session_id = store.event(
            "session_start",
            {
                "budget": asdict(limits.budget),
                "cancellation_enabled": cancellation is not None,
                "execution_backend": backend,
                **(worker_pool._metadata(run_id) if worker_pool is not None else {}),
            },
        )
        reason = "completed"
        try:
            if limits.stop_reason():
                reason = limits.stop_reason()
                if worker_pool is not None:
                    worker_pool._stop(force=True)
            elif backend == "in_process":
                while coordinator.has_work:
                    if limits.admission_closed:
                        reason = "attempt_budget"
                        break
                    job = coordinator.dispatch()
                    if job is None:
                        break
                    limits.admitted += 1
                    trial, replication, _ = job
                    coordinator.accept(job, _evaluate(objective, trial.config, replication))
            else:
                reason = run_parallel(
                    coordinator, objective, limits, worker_pool=worker_pool, run_id=run_id
                )
        except BaseException as exc:
            coordinator.interrupt()
            store.event(
                "session_end",
                {
                    "session_id": session_id,
                    "termination_reason": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_seconds": limits.elapsed,
                    "attempts": store.total_attempts() - initial_attempts,
                    **(worker_pool._metadata(run_id) if worker_pool is not None else {}),
                },
            )
            raise
        snapshot = coordinator.snapshot()
        store.save(snapshot)
        session_attempts = store.total_attempts() - initial_attempts
        session_wall = limits.elapsed
        store.event(
            "session_end",
            {
                "session_id": session_id,
                "termination_reason": reason,
                "wall_seconds": session_wall,
                "attempts": session_attempts,
                **(worker_pool._metadata(run_id) if worker_pool is not None else {}),
            },
        )
        attempts, events = store.journal()
        unknown = sum(a["status"] != "accepted" for a in attempts)
        return StudyResult(
            coordinator.trials,
            coordinator.decisions,
            snapshot["wall_seconds"],
            {
                **manifest,
                "package_version": __version__,
                "termination_reason": reason,
                "session_id": session_id,
                "session_budget": asdict(limits.budget),
                "session_attempts": session_attempts,
                "session_wall_seconds": session_wall,
                "execution_backend": backend,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "persistent": study.storage is not None,
                "attempts": attempts,
                "events": events,
                "unmeasured_attempts": unknown,
                "objective_cpu_accounting_complete": unknown == 0,
                "native_thread_control": control,
                "allocated_worker_seconds": snapshot["wall_seconds"] * study.workers,
                **(worker_pool._metadata(run_id) if worker_pool is not None else {}),
            },
        )
