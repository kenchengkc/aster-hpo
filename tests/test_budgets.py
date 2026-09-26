import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest

from aster_hpo import CancellationToken, FullBudget, RunBudget, Study


def slow_or_fast(config, replication):
    marker = config.get("marker")
    if marker:
        Path(marker).write_text(str(os.getpid()))
    if config.get("slow"):
        time.sleep(60)
    return float(config["x"] + replication.index)


def pure(config, replication):
    return float(config["x"] + replication.index)


def crash_once(config, replication):
    try:
        Path(config["marker"]).touch(exist_ok=False)
    except FileExistsError:
        return pure(config, replication)
    os._exit(17)


def assert_workers_stopped():
    assert not [p for p in multiprocessing.active_children() if p.name.startswith("aster-worker")]


def test_parallel_attempt_cap_drains_exact_admitted_work_and_resumes(tmp_path):
    study = Study(
        [{"x": x} for x in range(8)],
        workers=3,
        scheduler=FullBudget(3),
        storage=tmp_path / "study.sqlite",
        objective_id="pure-v1",
    )
    first = study.optimize(pure, budget=RunBudget(max_attempts=5))
    assert first.metadata["termination_reason"] == "attempt_budget"
    assert first.metadata["session_attempts"] == 5
    assert sum(len(t.observations) for t in first.trials) == 5
    assert not any(t.status == "running" for t in first.trials)
    second = study.optimize(pure)
    assert [t.losses for t in second.trials] == [[x, x + 1, x + 2] for x in range(8)]
    assert second.metadata["session_attempts"] == 19
    assert_workers_stopped()


@pytest.mark.parametrize("workers", [1, 3])
def test_wall_deadline_terminates_live_workers_and_preserves_partial_winner(tmp_path, workers):
    study = Study(
        [{"x": 0}, {"x": 1, "slow": True, "marker": str(tmp_path / "started")}],
        workers=workers,
        scheduler=FullBudget(1),
        storage=tmp_path / "study.sqlite",
        objective_id="slow-fast-v1",
        max_retries=0,
    )
    started = time.perf_counter()
    result = study.optimize(slow_or_fast, budget=RunBudget(max_wall_seconds=1))
    assert time.perf_counter() - started < 5
    assert (tmp_path / "started").exists()
    assert result.metadata["termination_reason"] == "wall_budget"
    assert result.best_trial.config == {"x": 0}
    assert result.trials[1].status == "pending"
    assert result.trials[1].observations == []
    assert result.metadata["unmeasured_attempts"] == 1
    assert [a["status"] for a in result.metadata["attempts"]] == ["accepted", "cancelled"]
    assert_workers_stopped()


def test_caller_cancellation_can_resume_without_infrastructure_retries(tmp_path):
    token = CancellationToken()
    marker = tmp_path / "started"
    study = Study(
        [{"x": 1, "slow": True, "marker": str(marker)}],
        workers=2,
        scheduler=FullBudget(1),
        storage=tmp_path / "study.sqlite",
        objective_id="cancellation-fixture-v1",
        max_retries=0,
    )

    def cancel_after_start():
        end = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < end:
            time.sleep(0.01)
        token.cancel()

    thread = threading.Thread(target=cancel_after_start)
    thread.start()
    first = study.optimize(slow_or_fast, cancellation=token)
    thread.join(timeout=1)
    assert marker.exists()
    assert first.metadata["termination_reason"] == "cancelled"
    assert first.best_trial is None
    assert_workers_stopped()
    # Same objective contract; this fixture's resumed evaluator completes the
    # same deterministic value without repeating the artificial blocking delay.
    second = study.optimize(pure)
    assert second.best_trial.mean == 1
    assert (
        second.metadata["attempts"][0]["replication"]
        == second.metadata["attempts"][1]["replication"]
    )
    assert second.metadata["unmeasured_attempts"] == 1
    assert_workers_stopped()


def test_evaluation_timeout_fails_only_its_trial_then_reuses_capacity(tmp_path):
    result = Study([{"x": 0, "slow": True}, {"x": 1}], workers=1, scheduler=FullBudget(2)).optimize(
        slow_or_fast, budget=RunBudget(evaluation_timeout_seconds=0.7)
    )
    assert result.metadata["termination_reason"] == "completed"
    assert result.trials[0].status == "failed"
    assert result.trials[0].observations == []
    assert result.best_trial.losses == [1, 2]
    assert result.metadata["attempts"][0]["status"] == "timed_out"
    assert result.metadata["unmeasured_attempts"] == 1
    assert_workers_stopped()


def test_worker_retry_uses_attempt_allowance(tmp_path):
    study = Study(
        [{"x": 0, "marker": str(tmp_path / "crashed")}],
        workers=2,
        scheduler=FullBudget(1),
        storage=tmp_path / "study.sqlite",
        objective_id="crash-once-v1",
        max_retries=1,
    )
    first = study.optimize(crash_once, budget=RunBudget(max_attempts=1))
    assert first.metadata["termination_reason"] == "attempt_budget"
    assert first.metadata["session_attempts"] == 1
    assert first.trials[0].status == "pending"
    second = study.optimize(crash_once, budget=RunBudget(max_attempts=1))
    assert second.best_trial.mean == 0
    assert second.metadata["session_attempts"] == 1
    assert_workers_stopped()


def straggling_pruned(config, replication):
    if config["x"] == 0 and replication.index == 0:
        time.sleep(0.3)
    if config["x"] == 10 and replication.index == 2:
        time.sleep(1)
    return config["x"]


def test_already_running_pruned_work_is_drained_after_last_survivor_completes():
    from aster_hpo import ReplicationRace

    result = Study([{"x": 10}, {"x": 0}], workers=2, scheduler=ReplicationRace((2, 3))).optimize(
        straggling_pruned
    )
    assert result.trials[0].status == "pruned"
    assert len(result.trials[0].observations) == 3
    assert all(a["status"] == "accepted" for a in result.metadata["attempts"])
    assert result.metadata["unmeasured_attempts"] == 0
    assert_workers_stopped()


def ignore_termination(config, replication):
    import signal

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    return slow_or_fast(config, replication)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal-resistance fixture")
def test_cancellation_kills_a_worker_that_ignores_termination(tmp_path):
    token = CancellationToken()
    marker = tmp_path / "started"

    def cancel_when_ready():
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        token.cancel()

    sender = threading.Thread(target=cancel_when_ready)
    sender.start()
    started = time.perf_counter()
    result = Study([{"x": 0, "slow": True, "marker": str(marker)}]).optimize(
        ignore_termination, budget=RunBudget(max_wall_seconds=6), cancellation=token
    )
    sender.join(timeout=1)
    assert marker.exists()
    assert time.perf_counter() - started < 5
    assert result.metadata["termination_reason"] == "cancelled"
    assert result.metadata["attempts"][0]["status"] == "cancelled"
    assert_workers_stopped()
    assert not [t for t in threading.enumerate() if t.name == "aster-worker-io"]


class FailsSecondSerialization:
    def __init__(self):
        self.calls = 0

    def __reduce__(self):
        self.calls += 1
        if self.calls == 2:
            raise TypeError("fixture cannot pickle second worker")
        return FailsSecondSerialization, ()

    def __call__(self, config, replication):
        time.sleep(60)
        return 0


def test_serialization_error_reaps_previously_started_workers_and_keeps_journal(tmp_path):
    import sqlite3

    database = tmp_path / "study.sqlite"
    study = Study(
        [{"x": 0}, {"x": 1}],
        workers=2,
        scheduler=FullBudget(1),
        storage=database,
        objective_id="serialization-v1",
    )
    with pytest.raises(TypeError, match="pickle second worker"):
        study.optimize(FailsSecondSerialization())
    assert_workers_stopped()
    assert not [t for t in threading.enumerate() if t.name == "aster-worker-io"]
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM attempts WHERE status='running'").fetchone()[0]
            == 0
        )
    result = study.optimize(pure)
    assert result.best_trial.mean == 0
    assert [t.status for t in result.trials] == ["complete", "complete"]


def test_timed_single_worker_requires_a_picklable_objective():
    with pytest.raises((AttributeError, TypeError), match="pickle"):
        Study([{"x": 0}]).optimize(lambda c, r: 0, budget=RunBudget(max_wall_seconds=1))
    assert_workers_stopped()
