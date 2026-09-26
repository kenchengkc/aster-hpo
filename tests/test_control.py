import json

import pytest

from aster_hpo import FullBudget, Study


def value(config, replication):
    return config["x"] + replication.index


@pytest.mark.parametrize(
    "settings",
    [
        {"max_attempts": 0},
        {"max_attempts": True},
        {"max_attempts": 1.5},
        {"max_wall_seconds": 0},
        {"max_wall_seconds": float("nan")},
        {"max_wall_seconds": True},
        {"evaluation_timeout_seconds": -1},
        {"evaluation_timeout_seconds": float("inf")},
    ],
)
def test_budget_rejects_invalid_limits(settings):
    from aster_hpo import RunBudget

    with pytest.raises(ValueError):
        RunBudget(**settings)


def test_serial_attempt_cap_resumes_with_a_new_session_allowance(tmp_path):
    from aster_hpo import RunBudget

    study = Study(
        [{"x": 0}, {"x": 10}],
        scheduler=FullBudget(3),
        storage=tmp_path / "study.sqlite",
        objective_id="value-v1",
        max_retries=0,
    )
    first = study.optimize(value, budget=RunBudget(max_attempts=2))
    assert first.metadata["termination_reason"] == "attempt_budget"
    assert first.metadata["session_attempts"] == 2
    assert [t.losses for t in first.trials] == [[0], [10]]
    assert first.best_trial is None
    second = study.optimize(value, budget=RunBudget(max_attempts=4))
    assert second.metadata["termination_reason"] == "completed"
    assert second.metadata["session_attempts"] == 4
    assert [t.losses for t in second.trials] == [[0, 1, 2], [10, 11, 12]]
    assert second.best_trial.id == 0
    starts = [e for e in second.metadata["events"] if e["kind"] == "session_start"]
    ends = [e for e in second.metadata["events"] if e["kind"] == "session_end"]
    assert [e["payload"]["budget"]["max_attempts"] for e in starts] == [2, 4]
    assert [e["payload"]["termination_reason"] for e in ends] == ["attempt_budget", "completed"]
    assert [e["payload"]["session_id"] for e in ends] == [e["seq"] for e in starts]
    second.save(tmp_path / "export.json")
    assert json.loads((tmp_path / "export.json").read_text())["metadata"]["session_attempts"] == 4


def test_pre_cancelled_session_starts_no_evaluation(tmp_path):
    from aster_hpo import CancellationToken

    token = CancellationToken()
    token.cancel()
    token.cancel()
    assert token.is_cancelled
    study = Study(
        [{"x": 0}],
        scheduler=FullBudget(1),
        storage=tmp_path / "study.sqlite",
        objective_id="value-v1",
    )
    result = study.optimize(value, cancellation=token)
    assert result.metadata["termination_reason"] == "cancelled"
    assert result.metadata["session_attempts"] == 0
    assert result.trials[0].status == "pending"
    assert study.optimize(value).best_trial.losses == [0]


def test_cancelled_attempts_count_as_work_but_not_infrastructure_retries(tmp_path):
    from aster_hpo.core import Replication
    from aster_hpo.runtime import Coordinator
    from aster_hpo.storage import SQLiteStore

    with SQLiteStore(tmp_path / "study.sqlite", {}) as store:
        study = Study([{"x": 0}], scheduler=FullBudget(1), max_retries=0)
        coordinator = Coordinator(study, store)
        job = coordinator.dispatch()
        store.stop_attempts([job[2]], "cancelled", "wall budget", coordinator.snapshot())
        assert store.total_attempts() == 1
        assert store.attempt_count(0, 0) == 0
        resumed = Coordinator(study, store)
        assert resumed.dispatch()[1] == Replication.create(0, {"x": 0}, 0, "search", True)


def test_unbounded_one_worker_keeps_local_callable_support():
    study = Study([{"x": 0}], scheduler=FullBudget(1))
    result = study.optimize(lambda config, replication: 3)
    assert result.best_trial.mean == 3
    assert result.metadata["execution_backend"] == "in_process"
