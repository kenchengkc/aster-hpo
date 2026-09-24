import os
import random
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from aster_hpo import FullBudget, Study


def deterministic(config, replication):
    return config["x"] + random.Random(replication.scenario_seed).random()


def crash_once(config, replication):
    marker = Path(config["marker"])
    try:
        marker.touch(exist_ok=False)
    except FileExistsError:
        pass
    else:
        os._exit(19)
    return deterministic(config, replication)


def crash_always(config, replication):
    os._exit(19)


def worker_thread_setting(config, replication):
    return float(os.environ["OMP_NUM_THREADS"])


def test_interrupt_resume_preserves_accepted_observations(tmp_path):
    db = tmp_path / "run.sqlite"
    calls = []
    interrupted = False

    def objective(config, replication):
        nonlocal interrupted
        calls.append(replication.index)
        if replication.index == 2 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return deterministic(config, replication)

    settings = dict(scheduler=FullBudget(4), storage=db, objective_id="fixture-v1")
    with pytest.raises(KeyboardInterrupt):
        Study([{"x": 1}], **settings).optimize(objective)
    result = Study([{"x": 1}], **settings).optimize(objective)
    assert calls == [0, 1, 2, 2, 3]
    reference = Study([{"x": 1}], scheduler=FullBudget(4)).optimize(deterministic)
    assert result.trials[0].losses == reference.trials[0].losses
    assert result.metadata["unmeasured_attempts"] == 1
    assert not result.metadata["objective_cpu_accounting_complete"]
    Study([{"x": 1}], **settings).optimize(objective)
    assert calls == [0, 1, 2, 2, 3]
    with pytest.raises(ValueError, match="manifest"):
        Study([{"x": 2}], **settings).optimize(objective)
    assert calls == [0, 1, 2, 2, 3]


def test_worker_death_retries_same_replication_and_records_cost_gap(tmp_path):
    config = {"x": 1, "marker": str(tmp_path / "crashed")}
    result = Study(
        [config],
        workers=2,
        scheduler=FullBudget(3),
        max_retries=1,
        storage=tmp_path / "run.sqlite",
        objective_id="crash-once-v1",
    ).optimize(crash_once)
    reference = Study([config], scheduler=FullBudget(3)).optimize(deterministic)
    assert result.trials[0].losses == reference.trials[0].losses
    assert result.metadata["unmeasured_attempts"] == 1
    first_two = result.metadata["attempts"][:2]
    assert first_two[0]["replication"] == first_two[1]["replication"]
    assert first_two[0]["status"] == "infrastructure_failed"
    assert first_two[1]["status"] == "accepted"


def test_worker_retries_are_bounded(tmp_path):
    db = tmp_path / "bounded.sqlite"
    with pytest.raises(RuntimeError, match="retry budget"):
        Study([{"x": 1}], workers=2, max_retries=1, storage=db, objective_id="crash-v1").optimize(
            crash_always
        )
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2


def test_numerical_failure_is_not_retried():
    def objective(config, replication):
        return float("nan")

    result = Study([{"x": 0}], max_retries=3).optimize(objective)
    assert result.best_trial is None
    assert len(result.metadata["attempts"]) == 1


def test_thread_environment_is_restored(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    result = Study([{"x": 0}], workers=2, scheduler=FullBudget(2), threads_per_worker=2).optimize(
        worker_thread_setting
    )
    assert result.trials[0].losses == [2.0, 2.0]
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert result.trials[0].observations[0].worker_pid != os.getpid()


def test_persistent_studies_require_explicit_objective_identity(tmp_path):
    with pytest.raises(ValueError, match="objective_id"):
        Study([{"x": 0}], storage=tmp_path / "run.sqlite")


def test_audit_is_resumable_and_keeps_its_seed_namespace(tmp_path):
    from aster_hpo import audit

    calls = []

    def objective(config, replication):
        calls.append((replication.phase, replication.index))
        return deterministic(config, replication)

    settings = dict(
        replications=3, seed=7, storage=tmp_path / "audit.sqlite", objective_id="audit-v1"
    )
    first = audit(objective, {"x": 0}, **settings)
    second = audit(objective, {"x": 0}, **settings)
    assert calls == [("audit", 0), ("audit", 1), ("audit", 2)]
    assert first.mean == second.mean
    assert second.metadata["phase"] == "audit"
    assert second.wall_seconds > 0


def test_killed_coordinator_resumes_without_repeating_committed_work(tmp_path):
    driver = tmp_path / "driver.py"
    db = tmp_path / "killed.sqlite"
    driver.write_text("""
import sys, time
from pathlib import Path
from aster_hpo import Study, FullBudget
root = Path(sys.argv[1])
def objective(config, replication):
    with (root / 'calls.txt').open('a') as handle:
        handle.write(str(replication.index) + '\\n')
    if replication.index == 2:
        (root / 'ready').touch()
        time.sleep(30)
    return float(replication.index)
Study([{'x': 0}], scheduler=FullBudget(4), storage=root / 'killed.sqlite',
      objective_id='kill-demo-v1').optimize(objective)
""")
    process = subprocess.Popen([sys.executable, str(driver), str(tmp_path)])
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail("coordinator exited before interruption point")
            time.sleep(0.01)
        assert (tmp_path / "ready").exists()
        process.kill()
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    calls = []

    def resumed(config, replication):
        calls.append(replication.index)
        return float(replication.index)

    result = Study(
        [{"x": 0}], scheduler=FullBudget(4), storage=db, objective_id="kill-demo-v1"
    ).optimize(resumed)
    assert calls == [2, 3]
    assert result.trials[0].losses == [0, 1, 2, 3]
