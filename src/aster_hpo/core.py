"""Asynchronous replication allocation. Statistical pruning is a heuristic."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import platform
import statistics
import tempfile
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        loss = float(objective(dict(config), replication))
        if not math.isfinite(loss):
            raise ValueError("Objective returned a non-finite loss")
        error = None
    except Exception as exc:
        loss, error = None, f"{type(exc).__name__}: {exc}"
    return Observation(
        replication, loss, time.perf_counter() - wall, time.process_time() - cpu, error
    )


def _pruning_decision(trial: Trial, trials: list[Trial], policy: ReplicationRace) -> dict | None:
    n = len(trial.losses)
    if n not in policy.checkpoints[:-1]:
        return None
    peers = [
        t
        for t in trials
        if t.id != trial.id and t.status not in {"failed", "pruned"} and len(t.losses) >= n
    ]
    if not peers:
        return None
    peer = min(peers, key=lambda t: (statistics.mean(t.losses[:n]), t.id))
    differences = [a - b for a, b in zip(trial.losses, peer.losses[:n], strict=True)]
    mean = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(n)
    lower = mean - policy.z * se
    return {
        "trial_id": trial.id,
        "peer_id": peer.id,
        "replications": n,
        "mean_difference": mean,
        "standard_error": se,
        "heuristic_lower_bound": lower,
        "tolerance": policy.tolerance,
        "action": "prune" if lower > policy.tolerance else "continue",
    }


class Study:
    """Minimize a finite configuration pool with one in-flight replication per trial.

    Workers use spawn, so objectives must be importable top-level callables. A one-worker
    study runs in the caller for debugging. No retries, deadlines, or resume in v0.1.
    """

    def __init__(
        self,
        configs: Sequence[Mapping[str, Scalar]],
        *,
        scheduler: FullBudget | ReplicationRace | None = None,
        workers: int = 1,
        seed: int = 0,
        shared_scenarios: bool = True,
    ):
        _positive_integer(workers, "workers")
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
        if not isinstance(self.scheduler, FullBudget | ReplicationRace):
            raise TypeError("Unsupported scheduler")
        self.workers, self.seed, self.shared_scenarios = workers, seed, shared_scenarios

    def optimize(self, objective: Objective) -> StudyResult:
        start = time.perf_counter()
        trials = [Trial(i, dict(c)) for i, c in enumerate(self.configs)]
        ready = deque(range(len(trials)))
        decisions = []

        def replication_for(trial):
            return Replication.create(
                self.seed, trial.config, len(trial.observations), "search", self.shared_scenarios
            )

        def accept(trial, observation):
            trial.observations.append(observation)
            if observation.error is not None:
                trial.status = "failed"
                return
            if len(trial.losses) == self.scheduler.maximum:
                trial.status = "complete"
                return
            if isinstance(self.scheduler, ReplicationRace):
                decision = _pruning_decision(trial, trials, self.scheduler)
                if decision:
                    decisions.append(decision)
                    if decision["action"] == "prune":
                        trial.status = "pruned"
                        return
            trial.status = "pending"
            ready.append(trial.id)

        if self.workers == 1:
            while ready:
                trial = trials[ready.popleft()]
                accept(trial, _evaluate(objective, trial.config, replication_for(trial)))
        else:
            with ProcessPoolExecutor(
                max_workers=self.workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                pending = {}
                while ready or pending:
                    while ready and len(pending) < self.workers:
                        trial = trials[ready.popleft()]
                        trial.status = "running"
                        future = pool.submit(
                            _evaluate, objective, trial.config, replication_for(trial)
                        )
                        pending[future] = trial
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in sorted(done, key=lambda f: pending[f].id):
                        # Broken pools / serialization errors abort rather than fabricate losses.
                        accept(pending.pop(future), future.result())

        return StudyResult(
            trials,
            decisions,
            time.perf_counter() - start,
            {
                "schema_version": 1,
                "package_version": "0.1.0",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "workers": self.workers,
                "seed": self.seed,
                "shared_scenarios": self.shared_scenarios,
                "scheduler": type(self.scheduler).__name__,
                "scheduler_parameters": asdict(self.scheduler),
            },
        )


@dataclass
class AuditResult:
    observations: list[Observation]

    @property
    def mean(self) -> float:
        return statistics.mean(o.loss for o in self.observations)

    @property
    def standard_error(self) -> float:
        return statistics.stdev(o.loss for o in self.observations) / math.sqrt(
            len(self.observations)
        )


def audit(
    objective: Objective, config: Config, *, replications: int = 30, seed: int = 0
) -> AuditResult:
    """Evaluate a locked configuration on an isolated seed namespace, serially.

    Choose sample count in advance; mean and standard error are descriptive. Repeated
    calls with the same inputs reuse data and must not be treated as fresh evidence.
    """
    _positive_integer(replications, "replications")
    if replications < 2:
        raise ValueError("Audit requires at least two replications")
    observations = []
    for index in range(replications):
        observation = _evaluate(
            objective, config, Replication.create(seed, config, index, "audit", True)
        )
        if observation.error is not None:
            raise RuntimeError(f"Audit replication {index} failed: {observation.error}")
        observations.append(observation)
    return AuditResult(observations)
