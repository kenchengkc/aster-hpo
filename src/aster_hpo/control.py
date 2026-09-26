"""Per-invocation resource allowances and caller cancellation."""

import math
import time
from dataclasses import dataclass
from threading import Event


@dataclass(frozen=True)
class RunBudget:
    """Limits for one optimize invocation, not lifetime limits on a saved study.

    max_attempts includes retries and drains admitted work. Timed limits preempt
    managed workers; process cleanup and coordinator overhead add stopping latency.
    Evaluation timeouts start at dispatch, including worker startup and IPC.
    """

    max_wall_seconds: float | None = None
    max_attempts: int | None = None
    evaluation_timeout_seconds: float | None = None

    def __post_init__(self):
        for name in ("max_wall_seconds", "evaluation_timeout_seconds"):
            value = getattr(self, name)
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.max_attempts is not None and (
            type(self.max_attempts) is not int or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")


class CancellationToken:
    """Thread-safe, one-way caller cancellation; use a new token to resume."""

    def __init__(self):
        self._event = Event()

    def cancel(self):
        self._event.set()

    @property
    def is_cancelled(self):
        return self._event.is_set()


class SessionControl:
    def __init__(self, budget, cancellation):
        if budget is not None and not isinstance(budget, RunBudget):
            raise TypeError("budget must be a RunBudget or None")
        if cancellation is not None and not isinstance(cancellation, CancellationToken):
            raise TypeError("cancellation must be a CancellationToken or None")
        self.budget = budget if budget is not None else RunBudget()
        self.cancellation = cancellation
        self.started = time.perf_counter()
        self.admitted = 0

    @property
    def elapsed(self):
        return time.perf_counter() - self.started

    @property
    def needs_process(self):
        return (
            self.budget.max_wall_seconds is not None
            or self.budget.evaluation_timeout_seconds is not None
            or self.cancellation is not None
        )

    @property
    def admission_closed(self):
        return self.budget.max_attempts is not None and self.admitted >= self.budget.max_attempts

    def stop_reason(self):
        if self.cancellation is not None and self.cancellation.is_cancelled:
            return "cancelled"
        if (
            self.budget.max_wall_seconds is not None
            and self.elapsed >= self.budget.max_wall_seconds
        ):
            return "wall_budget"
        return None
