"""Asynchronous replication allocation. Statistical pruning is a heuristic."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .control import CancellationToken, RunBudget
from .schedulers import ASHA
from .space import Scalar

Config = dict[str, Scalar]


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _seed(root: int, phase: str, index: int, stream: str, identity: str) -> int:
    raw = json.dumps([root, phase, index, stream, identity], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big")


@dataclass(frozen=True)
class Replication:
    """Stable seeds; shared scenarios omit configuration identity only from scenarios."""

    index: int
    scenario_seed: int
    training_seed: int
    phase: str

    @classmethod
    def create(cls, root: int, config: Config, index: int, phase: str, shared: bool):
        identity = json.dumps(config, sort_keys=True, allow_nan=False)
        return cls(
            index,
            _seed(root, phase, index, "scenario", "shared" if shared else identity),
            _seed(root, phase, index, "training", identity),
            phase,
        )


Objective = Callable[[Config, Replication], float]


@dataclass(frozen=True)
class FullBudget:
    replications: int = 27

    def __post_init__(self):
        _positive_integer(self.replications, "replications")

    @property
    def maximum(self) -> int:
        return self.replications


@dataclass(frozen=True)
class ReplicationRace:
    """Prune when mean paired disadvantage - z * standard error > tolerance.

    Repeated inspection and incumbent selection invalidate a nominal CI interpretation.
    This rule provides no selection or non-inferiority guarantee.
    """

    checkpoints: tuple[int, ...] = (3, 9, 27)
    z: float = 2.0
    tolerance: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))
        if (
            not self.checkpoints
            or any(type(n) is not int or n < 2 for n in self.checkpoints)
            or tuple(sorted(set(self.checkpoints))) != self.checkpoints
        ):
            raise ValueError("checkpoints must be strictly increasing integers >= 2")
        if not math.isfinite(self.z) or self.z <= 0:
            raise ValueError("z must be finite and positive")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")

    @property
    def maximum(self) -> int:
        return self.checkpoints[-1]


@dataclass
class Observation:
    replication: Replication
    loss: float | None
    wall_seconds: float
    cpu_seconds: float
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    worker_pid: int | None = None


@dataclass
class Trial:
    id: int
    config: Config
    status: str = "pending"
    observations: list[Observation] = field(default_factory=list)

    @property
    def losses(self) -> list[float]:
        return [o.loss for o in self.observations if o.loss is not None]

    @property
    def mean(self) -> float | None:
        return statistics.mean(self.losses) if self.losses else None


@dataclass
class StudyResult:
    trials: list[Trial]
    decisions: list[dict]
    wall_seconds: float
    metadata: dict

    @property
    def best_trial(self) -> Trial | None:
        # Compare only unpruned configurations evaluated to the common resource cap.
        complete = [t for t in self.trials if t.status == "complete"]
        return min(complete, key=lambda t: (t.mean, t.id)) if complete else None

    @property
    def cpu_seconds(self) -> float:
        """Sum objective process CPU time, excluding parent and process startup."""
        return sum(o.cpu_seconds for t in self.trials for o in t.observations)

    def save(self, path: str | Path) -> None:
        """Atomically save a completed result; this is not a resumable checkpoint."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["best_trial_id"] = self.best_trial.id if self.best_trial else None
        payload["objective_cpu_seconds"] = self.cpu_seconds
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=destination.parent, encoding="utf-8", delete=False
            ) as handle:
                temp_name = handle.name
                json.dump(payload, handle, indent=2, allow_nan=False)
                handle.write("\n")
            os.replace(temp_name, destination)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)


def _evaluate(objective: Objective, config: Config, replication: Replication) -> Observation:
    started_at = time.time()
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        loss = float(objective(dict(config), replication))
        if not math.isfinite(loss):
            raise ValueError("Objective returned a non-finite loss")
        error = None
    except Exception as exc:
        loss, error = None, f"{type(exc).__name__}: {exc}"
    return Observation(
        replication,
        loss,
        time.perf_counter() - wall,
        time.process_time() - cpu,
        error,
        started_at,
        time.time(),
        os.getpid(),
    )


class Study:
    """Minimize a finite configuration pool with one in-flight replication per trial.

    Workers use spawn, so objectives must be importable top-level callables. A one-worker
    study runs in the caller. Optional SQLite storage resumes committed progress.
    Infrastructure retries reuse seeds; objective failures are never retried.
    """

    def __init__(
        self,
        configs: Sequence[Mapping[str, Scalar]],
        *,
        scheduler: FullBudget | ReplicationRace | ASHA | None = None,
        workers: int = 1,
        seed: int = 0,
        shared_scenarios: bool = True,
        storage: str | Path | None = None,
        objective_id: str | None = None,
        max_retries: int = 1,
        threads_per_worker: int = 1,
    ):
        _positive_integer(workers, "workers")
        _positive_integer(threads_per_worker, "threads_per_worker")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if storage is not None and (not isinstance(objective_id, str) or not objective_id.strip()):
            raise ValueError("Persistent studies require an explicit objective_id")
        self.storage, self.objective_id = storage, objective_id
        self._phase = "search"
        self.max_retries, self.threads_per_worker = max_retries, threads_per_worker
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if not configs:
            raise ValueError("At least one configuration is required")
        for config in configs:
            if any(not isinstance(k, str) for k in config):
                raise ValueError("Configuration keys must be strings")
            if any(type(v) not in {str, int, float, bool, type(None)} for v in config.values()):
                raise ValueError("Configuration values must be JSON scalars")
        self.configs = json.loads(json.dumps([dict(c) for c in configs], allow_nan=False))
        self.scheduler = scheduler if scheduler is not None else FullBudget()
        if not isinstance(self.scheduler, FullBudget | ReplicationRace | ASHA):
            raise TypeError("Unsupported scheduler")
        self.workers, self.seed, self.shared_scenarios = workers, seed, shared_scenarios

    def optimize(
        self,
        objective: Objective,
        *,
        budget: RunBudget | None = None,
        cancellation: CancellationToken | None = None,
    ) -> StudyResult:
        from .runtime import run

        return run(self, objective, budget=budget, cancellation=cancellation)


@dataclass
class AuditResult:
    observations: list[Observation]
    wall_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return statistics.mean(o.loss for o in self.observations)

    @property
    def standard_error(self) -> float:
        return statistics.stdev(o.loss for o in self.observations) / math.sqrt(
            len(self.observations)
        )


def audit(
    objective: Objective,
    config: Config,
    *,
    replications: int = 30,
    seed: int = 0,
    workers: int = 1,
    storage: str | Path | None = None,
    objective_id: str | None = None,
    max_retries: int = 1,
    threads_per_worker: int = 1,
) -> AuditResult:
    """Evaluate a locked configuration in a separate, optionally durable namespace.

    Choose sample count in advance. Repeated calls with identical seeds reuse data
    and must not be treated as fresh evidence. A persistent audit resumes work.
    """
    _positive_integer(replications, "replications")
    if replications < 2:
        raise ValueError("Audit requires at least two replications")
    study = Study(
        [config],
        scheduler=FullBudget(replications),
        workers=workers,
        seed=seed,
        storage=storage,
        objective_id=objective_id,
        max_retries=max_retries,
        threads_per_worker=threads_per_worker,
    )
    study._phase = "audit"
    result = study.optimize(objective)
    observations = result.trials[0].observations
    for observation in observations:
        if observation.error is not None:
            raise RuntimeError(
                f"Audit replication {observation.replication.index} failed: {observation.error}"
            )
    return AuditResult(observations, result.wall_seconds, result.metadata)
