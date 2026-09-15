"""Small, dependency-free building blocks for stochastic optimization."""

from .core import (
    AuditResult,
    FullBudget,
    Replication,
    ReplicationRace,
    Study,
    StudyResult,
    audit,
)
from .space import Choice, Float, Int, sample_configs

__version__ = "0.1.0"
__all__ = [
    "AuditResult",
    "Choice",
    "Float",
    "FullBudget",
    "Int",
    "Replication",
    "ReplicationRace",
    "Study",
    "StudyResult",
    "audit",
    "sample_configs",
]
