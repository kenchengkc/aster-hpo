import pytest

from aster_hpo import ReplicationRace, Study
from aster_hpo.core import Observation, Replication, Trial


def noisy(config, replication):
    import random

    return config["x"] ** 2 + random.Random(replication.scenario_seed).gauss(0, 1)


@pytest.mark.parametrize("xs", [[0, 5, 10], [10, 5, 0]])
def test_candidate_order_uses_available_checkpoint_peers(xs):
    result = Study([{"x": x} for x in xs], scheduler=ReplicationRace((3, 9))).optimize(noisy)
    assert sum(len(t.observations) for t in result.trials) == 15
    assert result.best_trial.config == {"x": 0}


def trial(index, losses, status="pending"):
    config = {"x": index}
    return Trial(
        index,
        config,
        status,
        [
            Observation(Replication.create(0, config, i, "search", True), loss, 0, 0)
            for i, loss in enumerate(losses)
        ],
    )


def policy():
    from aster_hpo.schedulers import RacePolicy

    return RacePolicy(ReplicationRace((3, 9)))


def test_late_peer_and_unchanged_prefix_deduplication():
    scheduler = policy()
    candidates = [trial(0, [5, 5, 5]), trial(1, [9, 9, 9]), trial(2, [0, 0])]
    first = list(scheduler.decisions(candidates))
    assert all(d["trial_id"] != 2 for d in first)
    assert list(scheduler.decisions(candidates)) == []
    candidates[2] = trial(2, [0, 0, 0])
    later = list(scheduler.decisions(candidates))
    eliminated = {d["trial_id"] for d in later if d["action"] == "prune"}
    assert eliminated == {0, 1}
    assert all(d["replication_indices"] == [0, 1, 2] for d in later)
    assert list(scheduler.decisions(candidates)) == []


def test_pairing_uses_checkpoint_prefix_and_residual_variance():
    scheduler = policy()
    candidates = [trial(0, [4, 6, 8, -100]), trial(1, [0, 1, 2])]
    decision = next(d for d in scheduler.decisions(candidates) if d["trial_id"] == 0)
    assert decision["mean_difference"] == 5
    assert decision["standard_error"] == pytest.approx(1 / 3**0.5)
    assert decision["replications"] == 3
    assert decision["action"] == "prune"


@pytest.mark.parametrize("status", ["failed", "pruned"])
def test_ineligible_peers_cannot_prune_survivor(status):
    assert list(policy().decisions([trial(0, [8] * 3), trial(1, [0] * 3, status)])) == []


def test_complete_candidates_are_peers_but_never_pruned():
    decisions = list(policy().decisions([trial(0, [0] * 9, "complete"), trial(1, [5] * 3)]))
    assert len(decisions) == 1
    assert decisions[0]["trial_id"] == 1
    assert decisions[0]["action"] == "prune"


def test_decisions_survive_restoring_seen_comparisons():
    from aster_hpo.schedulers import RacePolicy

    candidates = [trial(0, [0] * 3), trial(1, [5] * 3)]
    decisions = list(policy().decisions(candidates))
    restored = RacePolicy(ReplicationRace((3, 9)), decisions)
    assert list(restored.decisions(candidates)) == []


def test_in_flight_result_cannot_reactivate_pruned_trial():
    """Drive completion order at the coordinator boundary, independent of executor."""
    from aster_hpo.core import _evaluate
    from aster_hpo.runtime import Coordinator
    from aster_hpo.storage import SQLiteStore

    study = Study([{"x": 5}, {"x": 0}], scheduler=ReplicationRace((3, 9)), workers=2)
    with SQLiteStore(None, {}) as store:
        coordinator = Coordinator(study, store)
        pending = []
        accepted_after_prune = False
        while coordinator.has_work or pending:
            while len(pending) < 2:
                job = coordinator.dispatch()
                if job is None:
                    break
                pending.append(job)
            # The bad trial starts its fourth replication before the good trial
            # supplies its third observation. Complete that in-flight work last.
            job = min(pending, key=lambda j: (j[0].id == 0 and j[1].index >= 3, j[0].id == 1))
            pending.remove(job)
            candidate, replication, _ = job
            was_pruned = candidate.status == "pruned"
            coordinator.accept(job, _evaluate(noisy, candidate.config, replication))
            if was_pruned:
                accepted_after_prune = True
                assert candidate.status == "pruned"
        bad, good = coordinator.trials
        assert accepted_after_prune
        assert bad.status == "pruned"
        assert len(bad.observations) == 4
        assert good.status == "complete"
        assert len(good.observations) == 9
        assert len(store.load()["trials"][0]["observations"]) == 4
        assert all(a["status"] == "accepted" for a in store.journal()[0])
