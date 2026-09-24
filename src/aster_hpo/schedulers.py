"""Checkpoint policies independent of worker execution.

Replication racing is a heuristic: adaptive comparisons do not supply a nominal
confidence or selection guarantee.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import ReplicationRace, Trial


@dataclass(frozen=True)
class ASHA:
    """Single-bracket stopping ASHA on replication-count resources.

    Rungs are geometric. At a rung, stop if loss exceeds the linearly interpolated
    1/reduction_factor quantile of previously reported losses; ties survive.
    This matches Ray 2.58.0's stopping bracket for finite losses, mode=min.
    """

    checkpoints: tuple[int, ...] = (3, 9, 27)
    reduction_factor: int = 3

    def __post_init__(self):
        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))
        if type(self.reduction_factor) is not int or self.reduction_factor < 2:
            raise ValueError("reduction_factor must be an integer >= 2")
        if not self.checkpoints or any(type(n) is not int or n < 1 for n in self.checkpoints):
            raise ValueError("checkpoints must be positive integers")
        if any(
            b != a * self.reduction_factor
            for a, b in zip(self.checkpoints[:-1], self.checkpoints[1:], strict=True)
        ):
            raise ValueError("ASHA checkpoints must be geometric")

    @property
    def maximum(self):
        return self.checkpoints[-1]


class ASHAPolicy:
    def __init__(self, settings, previous_decisions=()):
        self.settings = settings
        self.rungs = {n: {} for n in settings.checkpoints[:-1]}
        for decision in previous_decisions:
            self.rungs[decision["replications"]][decision["trial_id"]] = decision["mean_loss"]

    def decisions(self, trials):
        for trial in trials:
            n = len(trial.losses)
            if trial.status in {"pruned", "failed", "complete"} or n not in self.rungs:
                continue
            rung = self.rungs[n]
            if trial.id in rung:
                continue
            previous = sorted(rung.values())
            cutoff = None
            if previous:
                rank = (len(previous) - 1) / self.settings.reduction_factor
                low, high = math.floor(rank), math.ceil(rank)
                cutoff = previous[low] + (previous[high] - previous[low]) * (rank - low)
            score = statistics.mean(trial.losses)
            rung[trial.id] = score
            yield {
                "trial_id": trial.id,
                "replications": n,
                "replication_indices": list(range(n)),
                "mean_loss": score,
                "cutoff": cutoff,
                "prior_reports": len(previous),
                "action": "prune" if cutoff is not None and score > cutoff else "continue",
            }


class RacePolicy:
    """Reconsider matched prefixes when new comparison data becomes available."""

    def __init__(self, settings: ReplicationRace, previous_decisions=()):
        self.settings = settings
        self.seen = {(d["trial_id"], d["peer_id"], d["replications"]) for d in previous_decisions}

    def decisions(self, trials: list[Trial]):
        eligible = [t for t in trials if t.status not in {"failed", "pruned"}]
        losses = {t.id: t.losses for t in eligible}
        # Sort once per rung; status is checked again as callers apply yielded decisions.
        rungs = {
            n: sorted(
                [t for t in eligible if len(losses[t.id]) >= n],
                key=lambda t: (statistics.mean(losses[t.id][:n]), t.id),
            )
            for n in self.settings.checkpoints[:-1]
        }
        for trial in eligible:
            if trial.status in {"complete", "failed", "pruned"}:
                continue
            for n in reversed(self.settings.checkpoints[:-1]):
                if len(losses[trial.id]) < n:
                    continue
                peer = next(
                    (
                        t
                        for t in rungs[n]
                        if t.id != trial.id and t.status not in {"failed", "pruned"}
                    ),
                    None,
                )
                if peer is None:
                    continue
                key = (trial.id, peer.id, n)
                if key not in self.seen:
                    self.seen.add(key)
                    differences = [
                        a - b
                        for a, b in zip(losses[trial.id][:n], losses[peer.id][:n], strict=True)
                    ]
                    mean = statistics.mean(differences)
                    se = statistics.stdev(differences) / math.sqrt(n)
                    lower = mean - self.settings.z * se
                    yield {
                        "trial_id": trial.id,
                        "peer_id": peer.id,
                        "replications": n,
                        "replication_indices": list(range(n)),
                        "mean_difference": mean,
                        "standard_error": se,
                        "heuristic_lower_bound": lower,
                        "tolerance": self.settings.tolerance,
                        "action": "prune" if lower > self.settings.tolerance else "continue",
                    }
                # Never fall back to less information after a usable higher checkpoint.
                break
