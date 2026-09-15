"""Deterministic random configuration pools, independent of worker scheduling."""

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

Scalar = str | int | float | bool | None


class Distribution(Protocol):
    def sample(self, rng: random.Random) -> Scalar: ...


@dataclass(frozen=True)
class Float:
    low: float
    high: float
    log: bool = False

    def __post_init__(self):
        if not (math.isfinite(self.low) and math.isfinite(self.high) and self.low < self.high):
            raise ValueError("Float bounds must be finite and low < high")
        if self.log and self.low <= 0:
            raise ValueError("Log bounds must be positive")

    def sample(self, rng: random.Random) -> float:
        if self.log:
            return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        return rng.uniform(self.low, self.high)


@dataclass(frozen=True)
class Int:
    low: int
    high: int

    def __post_init__(self):
        if type(self.low) is not int or type(self.high) is not int or self.low > self.high:
            raise ValueError("Int bounds must be integers with low <= high")

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.low, self.high)


@dataclass(frozen=True)
class Choice:
    values: tuple[Scalar, ...]

    def __post_init__(self):
        if not self.values:
            raise ValueError("Choice requires at least one value")

    def sample(self, rng: random.Random) -> Scalar:
        return rng.choice(self.values)


def sample_configs(
    space: Mapping[str, Distribution], *, count: int, seed: int = 0
) -> list[dict[str, Scalar]]:
    """Sample with replacement. Int endpoints are inclusive; Float uses uniform()."""
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if not space:
        raise ValueError("space must not be empty")
    rng = random.Random(seed)
    return [{name: space[name].sample(rng) for name in sorted(space)} for _ in range(count)]
