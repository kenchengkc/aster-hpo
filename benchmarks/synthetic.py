"""Known-mean noisy objectives; no artificial delays or market interpretation."""

import math
import random
from dataclasses import dataclass

from aster_hpo import Float, sample_configs

CASES = ("additive", "heteroscedastic", "near_ties", "heavy_tail")


def pool(seed, count):
    return sample_configs({"x": Float(-2, 2)}, count=count, seed=seed)


def expected_loss(case, config):
    if case not in CASES:
        raise ValueError(f"Unknown synthetic case: {case}")
    return (0.02 if case == "near_ties" else 1.0) * config["x"] ** 2


def _student_five(rng):
    # Standardized t(5): finite variance 1 and heavy tails, exact mean 0.
    return rng.gauss(0, 1) / math.sqrt(rng.gammavariate(2.5, 2) / 3)


@dataclass(frozen=True)
class SyntheticObjective:
    case: str

    def __post_init__(self):
        if self.case not in CASES:
            raise ValueError(f"Unknown synthetic case: {self.case}")

    def __call__(self, config, replication):
        shared = random.Random(replication.scenario_seed)
        residual = random.Random(replication.training_seed)
        x = config["x"]
        mean = expected_loss(self.case, config)
        if self.case == "additive":
            return mean + shared.gauss(0, 1)
        if self.case == "heavy_tail":
            return mean + (0.3 + abs(x)) * _student_five(shared) + 0.2 * _student_five(residual)
        scale = 1 + 0.2 * x if self.case == "near_ties" else 0.3 + abs(x)
        return mean + scale * shared.gauss(0, 1) + 0.2 * residual.gauss(0, 1)
