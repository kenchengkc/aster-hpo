"""Checkpoint policies independent of worker execution.

Replication racing is a heuristic: adaptive comparisons do not supply a nominal
confidence or selection guarantee.
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import ReplicationRace, Trial


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
