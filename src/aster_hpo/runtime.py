"""Local coordinator, process execution, recovery, and thread controls."""

from __future__ import annotations

import multiprocessing
import os
import platform
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import asdict

from ._version import __version__
from .core import Observation, Replication, ReplicationRace, StudyResult, Trial, _evaluate
from .schedulers import ASHA, ASHAPolicy, RacePolicy
from .storage import SQLiteStore

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
                    if trial.id not in self.ready:
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
            trial = self.trials[self.ready.popleft()]
            if trial.status != "pending":
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

    def interrupt(self):
        attempts, _ = self.store.journal()
        for attempt in attempts:
            if attempt["status"] == "running":
                self.store.fail_attempt(
                    attempt["id"], "interrupted", "coordinator stopped", self.snapshot()
                )


def _parallel(coordinator, objective):
    pool = None
    pending = {}

    def restart(extra, error):
        nonlocal pool
        lost = list(extra)
        # Salvage already finished results before abandoning a broken pool.
        for future, job in list(pending.items()):
            if future.done():
                try:
                    observation = future.result()
                except BrokenProcessPool:
                    lost.append(job)
                else:
                    coordinator.accept(job, observation)
            else:
                lost.append(job)
        pending.clear()
        pool.shutdown(wait=True, cancel_futures=True)
        pool = None
        coordinator.retry(lost, error)

    try:
        while coordinator.ready or pending:
            if pool is None:
                pool = ProcessPoolExecutor(
                    max_workers=coordinator.study.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
            failed_submission = False
            while coordinator.ready and len(pending) < coordinator.study.workers:
                job = coordinator.dispatch()
                if job is None:
                    break
                trial, replication, _ = job
                try:
                    future = pool.submit(_evaluate, objective, trial.config, replication)
                except BrokenProcessPool as exc:
                    restart([job], exc)
                    failed_submission = True
                    break
                pending[future] = job
            if failed_submission:
                continue
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda f: pending[f][0].id):
                job = pending.pop(future)
                try:
                    observation = future.result()
                except BrokenProcessPool as exc:
                    restart([job], exc)
                    break
                coordinator.accept(job, observation)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def run(study, objective):
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
        try:
            if study.workers == 1:
                while job := coordinator.dispatch():
                    trial, replication, _ = job
                    coordinator.accept(job, _evaluate(objective, trial.config, replication))
            else:
                _parallel(coordinator, objective)
        except BaseException:
            coordinator.interrupt()
            raise
        snapshot = coordinator.snapshot()
        store.save(snapshot)
        attempts, events = store.journal()
        unknown = sum(a["status"] != "accepted" for a in attempts)
        return StudyResult(
            coordinator.trials,
            coordinator.decisions,
            snapshot["wall_seconds"],
            {
                **manifest,
                "package_version": __version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "persistent": study.storage is not None,
                "attempts": attempts,
                "events": events,
                "unmeasured_attempts": unknown,
                "objective_cpu_accounting_complete": unknown == 0,
                "native_thread_control": control,
                "allocated_worker_seconds": snapshot["wall_seconds"] * study.workers,
            },
        )
