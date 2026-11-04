"""Small, dependency-free building blocks for stochastic optimization."""

from ._version import __version__ as __version__
from .control import CancellationToken, RunBudget
from .core import (
    AuditResult,
    FullBudget,
    Replication,
    ReplicationRace,
    Study,
    StudyResult,
    audit,
)
from .schedulers import ASHA
from .space import Choice, Float, Int, sample_configs
from .workers import WorkerPool

__all__ = [
    "ASHA",
    "AuditResult",
    "Choice",
    "CancellationToken",
    "RunBudget",
    "Float",
    "FullBudget",
    "Int",
    "Replication",
    "ReplicationRace",
    "Study",
    "StudyResult",
    "WorkerPool",
    "audit",
    "sample_configs",
]
