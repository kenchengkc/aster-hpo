from types import SimpleNamespace

import pytest

from aster_hpo.core import Observation, Replication, Trial


def candidate(index, count, value):
    config = {"x": index}
    return Trial(
        index,
        config,
        "pending",
        [
            Observation(Replication.create(0, config, i, "search", True), value, 0, 0)
            for i in range(count)
        ],
    )


def test_asha_cutoff_uses_prior_rung_values_and_keeps_ties():
    from aster_hpo import ASHA
    from aster_hpo.schedulers import ASHAPolicy

    policy = ASHAPolicy(ASHA((3, 9), reduction_factor=3))
    first = list(policy.decisions([candidate(0, 3, 3)]))[0]
    assert first["cutoff"] is None and first["action"] == "continue"
    second = list(policy.decisions([candidate(1, 3, 6)]))[0]
    assert second["cutoff"] == 3 and second["action"] == "prune"
    third = list(policy.decisions([candidate(2, 3, 4)]))[0]
    assert third["cutoff"] == pytest.approx(4)
    assert third["action"] == "continue"
    assert list(policy.decisions([candidate(2, 3, 4)])) == []
    restored = ASHAPolicy(ASHA((3, 9), reduction_factor=3), [first, second, third])
    assert list(restored.decisions([candidate(2, 3, 4)])) == []


def test_asha_study_reaches_cap_with_a_valid_winner():
    from aster_hpo import ASHA, Study

    def objective(config, replication):
        return config["x"]

    result = Study([{"x": 0}, {"x": 5}, {"x": 10}], scheduler=ASHA((3, 9))).optimize(objective)
    assert result.best_trial.config == {"x": 0}
    assert sum(len(t.observations) for t in result.trials) == 15
    assert all(len(t.observations) <= 9 for t in result.trials)


def test_asha_matches_pinned_ray_bracket():
    ray = pytest.importorskip("ray")
    if ray.__version__ != "2.58.0":
        pytest.skip("Reference comparison pins Ray 2.58.0")
    from ray.tune.schedulers.async_hyperband import _Bracket

    from aster_hpo import ASHA
    from aster_hpo.schedulers import ASHAPolicy

    reference = _Bracket(min_t=3, max_t=27, reduction_factor=3, s=0)
    policy = ASHAPolicy(ASHA((3, 9, 27)))
    for index, count, value in [(0, 3, 3), (1, 3, 6), (2, 3, 1), (0, 9, 3), (2, 9, 1), (3, 3, 4)]:
        ours = list(policy.decisions([candidate(index, count, value)]))[0]["action"]
        theirs = reference.on_result(SimpleNamespace(trial_id=str(index)), count, -value)
        assert (ours == "prune") == (theirs == "STOP")


@pytest.mark.parametrize("checkpoints, factor", [((3, 8), 3), ((3, 9), 1), ((0, 3), 3)])
def test_asha_validates_geometric_rungs(checkpoints, factor):
    from aster_hpo import ASHA

    with pytest.raises(ValueError):
        ASHA(checkpoints, reduction_factor=factor)
