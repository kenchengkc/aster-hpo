import pytest

from aster_hpo import Replication


def state():
    return {
        "trials": [{"id": 0, "config": {"x": 0}, "status": "running", "observations": []}],
        "ready": [],
        "decisions": [],
        "wall_seconds": 0,
    }


def test_store_rejects_stale_and_duplicate_acceptance(tmp_path):
    from aster_hpo.storage import SQLiteStore

    replication = Replication.create(0, {"x": 0}, 0, "search", True)
    from dataclasses import asdict

    rep = asdict(replication)
    observation = {
        "replication": rep,
        "loss": 1,
        "wall_seconds": 0.1,
        "cpu_seconds": 0.1,
        "error": None,
    }
    with SQLiteStore(tmp_path / "run.sqlite", {"version": 1}) as store:
        old = store.begin_attempt(0, rep, state())
        store.fail_attempt(old, "interrupted", "fixture", state())
        current = store.begin_attempt(0, rep, state())
        assert not store.accept(old, observation, state())
        assert store.accept(current, observation, state())
        assert not store.accept(current, observation, state())
        loaded = store.load()
        assert len(loaded["trials"][0]["observations"]) == 1
        assert loaded["trials"][0]["observations"][0]["loss"] == 1


def test_database_has_one_coordinator_and_manifest_is_immutable(tmp_path):
    from aster_hpo.storage import SQLiteStore

    db = tmp_path / "run.sqlite"
    with SQLiteStore(db, {"version": 1}):
        with pytest.raises(RuntimeError, match="in use"):
            SQLiteStore(db, {"version": 1})
    with pytest.raises(ValueError, match="manifest"):
        SQLiteStore(db, {"version": 2})
    with SQLiteStore(db, {"version": 1}) as store:
        assert store.load() is None
