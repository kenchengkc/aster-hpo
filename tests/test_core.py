import json
import math
import random
from types import MappingProxyType

import pytest

from aster_hpo import FullBudget, Replication, ReplicationRace, Study, audit


def noisy(config, replication):
    return config["x"] ** 2 + random.Random(replication.scenario_seed).gauss(0, 1)


def fails(config, replication):
    if config["x"] == 1:
        raise RuntimeError("deliberate failure")
    return config["x"]


def nonfinite(config, replication):
    return float("nan")


def mutate(config, replication):
    config["x"] += 1
    return config["x"]


def test_seeds_shared_only_where_requested():
    a = Replication.create(7, {"x": 1}, 0, "search", True)
    b = Replication.create(7, {"x": 2}, 0, "search", True)
    independent = Replication.create(7, {"x": 2}, 0, "search", False)
    assert a.scenario_seed == b.scenario_seed
    assert a.training_seed != b.training_seed
    assert independent.scenario_seed != b.scenario_seed
    assert a == Replication.create(7, {"x": 1}, 0, "search", True)
    assert a.scenario_seed != Replication.create(7, {"x": 1}, 1, "search", True).scenario_seed


def test_full_budget_parallel_matches_serial():
    configs = [{"x": x} for x in [0, 1, 2]]
    serial = Study(configs, scheduler=FullBudget(5), workers=1, seed=19).optimize(noisy)
    parallel = Study(configs, scheduler=FullBudget(5), workers=2, seed=19).optimize(noisy)
    assert [t.losses for t in serial.trials] == [t.losses for t in parallel.trials]
    assert all(t.status == "complete" for t in parallel.trials)
    assert sum(len(t.observations) for t in parallel.trials) == 15
    assert parallel.best_trial.config == {"x": 0}


@pytest.mark.parametrize("workers", [1, 2])
def test_paired_pruning_retains_best_and_respects_budget(workers):
    result = Study(
        [{"x": 0}, {"x": 5}, {"x": 10}],
        scheduler=ReplicationRace((3, 9)),
        workers=workers,
    ).optimize(noisy)
    assert result.best_trial.config == {"x": 0}
    assert any(t.status == "pruned" for t in result.trials)
    assert sum(len(t.observations) for t in result.trials) < 27
    assert all(3 <= len(t.observations) <= 9 for t in result.trials)
    for decision in result.decisions:
        if decision["action"] == "prune":
            assert decision["heuristic_lower_bound"] > 0
            assert decision["replications"] == 3


def test_equal_candidates_are_not_pruned():
    result = Study([{"x": 1}] * 3, scheduler=ReplicationRace((3, 9))).optimize(noisy)
    assert all(t.status == "complete" for t in result.trials)
    assert len(result.decisions) > 0


def test_tolerance_keeps_small_disadvantages():
    result = Study([{"x": 0}, {"x": 1}], scheduler=ReplicationRace((3, 9), tolerance=2)).optimize(
        noisy
    )
    assert all(t.status == "complete" for t in result.trials)


@pytest.mark.parametrize("workers", [1, 2])
def test_failures_are_recorded_and_excluded(workers):
    result = Study([{"x": 1}, {"x": 2}], scheduler=FullBudget(3), workers=workers).optimize(fails)
    assert result.trials[0].status == "failed"
    assert len(result.trials[0].observations) == 1
    assert result.trials[0].observations[0].error == "RuntimeError: deliberate failure"
    assert result.best_trial.config == {"x": 2}


def test_no_winner_when_every_trial_fails():
    result = Study([{"x": 0}], scheduler=FullBudget(3)).optimize(nonfinite)
    assert result.best_trial is None
    assert result.trials[0].observations[0].loss is None
    assert "non-finite" in result.trials[0].observations[0].error


def test_audit_namespace_and_search_are_disjoint():
    config = {"x": 0}
    search = Study([config], scheduler=FullBudget(5), seed=4).optimize(noisy)
    checked = audit(noisy, config, replications=5, seed=4)
    search_seeds = {o.replication.scenario_seed for o in search.trials[0].observations}
    audit_seeds = {o.replication.scenario_seed for o in checked.observations}
    assert not search_seeds & audit_seeds
    assert len(audit_seeds) == 5
    assert all(o.replication.phase == "audit" for o in checked.observations)
    assert math.isfinite(checked.mean) and checked.standard_error > 0
    assert [o.loss for o in checked.observations] == [
        o.loss for o in audit(noisy, config, replications=5, seed=4).observations
    ]


def test_audit_failure_is_not_silently_dropped():
    with pytest.raises(RuntimeError, match="Audit replication 0 failed"):
        audit(nonfinite, {"x": 0})


def test_save_result_with_failure_is_valid_json(tmp_path):
    result = Study([{"x": 1}, {"x": 2}], scheduler=FullBudget(3)).optimize(fails)
    path = tmp_path / "nested" / "result.json"
    result.save(path)
    result.save(path)
    saved = json.loads(path.read_text())
    assert saved["best_trial_id"] == 1
    assert saved["trials"][0]["observations"][0]["loss"] is None
    assert saved["metadata"]["scheduler_parameters"] == {"replications": 3}
    assert saved["objective_cpu_seconds"] >= 0
    assert list(path.parent.iterdir()) == [path]


def test_objective_cannot_mutate_study_configs():
    config = {"x": 0}
    result = Study([config], scheduler=FullBudget(3)).optimize(mutate)
    assert config == {"x": 0}
    assert result.trials[0].config == config
    assert result.trials[0].losses == [1, 1, 1]


def test_read_only_mapping_supported():
    result = Study([MappingProxyType({"x": 0})], scheduler=FullBudget(2)).optimize(noisy)
    assert result.best_trial.config == {"x": 0}


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_budget(count):
    with pytest.raises(ValueError):
        FullBudget(count)


@pytest.mark.parametrize("points", [(), (1, 3), (3, 3), (9, 3), (3, 4.5)])
def test_invalid_race_checkpoints(points):
    with pytest.raises(ValueError):
        ReplicationRace(points)


@pytest.mark.parametrize("configs", [[], [{"x": float("nan")}], [{"x": []}], [{1: 2}]])
def test_invalid_configs(configs):
    with pytest.raises(ValueError):
        Study(configs)


def test_invalid_worker_count():
    with pytest.raises(ValueError):
        Study([{"x": 0}], workers=0)


def test_one_candidate_reaches_cap():
    result = Study([{"x": 1}], scheduler=ReplicationRace((3, 9))).optimize(noisy)
    assert result.trials[0].status == "complete"
    assert len(result.trials[0].observations) == 9


def test_audit_needs_two_samples():
    with pytest.raises(ValueError):
        audit(noisy, {"x": 0}, replications=1)
