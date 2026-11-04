import multiprocessing
import os
import pickle
import queue
import signal
import threading
import time
from pathlib import Path

import pytest

import aster_hpo as hpo


def pool_type():
    assert hasattr(hpo, "WorkerPool"), "WorkerPool must be public"
    return hpo.WorkerPool


def value(config, replication):
    if config.get("delay"):
        time.sleep(config["delay"])
    return float(config["x"] + replication.index)


class StatefulObjective:
    def __init__(self, offset, directory):
        self.offset, self.directory, self.calls = offset, Path(directory), 0

    def __call__(self, config, replication):
        self.calls += 1
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{replication.phase}-{replication.index}").write_text(str(os.getpid()))
        return self.offset + self.calls


def crash_once(config, replication):
    try:
        Path(config["marker"]).touch(exist_ok=False)
    except FileExistsError:
        return value(config, replication)
    os._exit(17)


def blocked(config, replication):
    if marker := config.get("marker"):
        Path(marker).write_text(str(os.getpid()))
    if config.get("block"):
        time.sleep(60)
    return value(config, replication)


def crash(config, replication):
    os._exit(17)


def thread_setting(config, replication):
    import numpy as np
    from threadpoolctl import threadpool_info

    np.ones((2, 2)) @ np.ones((2, 2))
    count = int(os.environ["OMP_NUM_THREADS"])
    assert all(pool["num_threads"] == count for pool in threadpool_info())
    return count


def assert_stopped():
    assert not [p for p in multiprocessing.active_children() if p.name.startswith("aster-worker")]
    assert not [t for t in threading.enumerate() if t.name == "aster-worker-io"]


def test_reuses_pid_with_fresh_callable_binding_for_each_search_and_audit(tmp_path):
    objective = StatefulObjective(10, tmp_path / "first")
    with pool_type()() as pool:
        assert pool.worker_pids == ()
        first = hpo.Study([{"x": 0}], scheduler=hpo.FullBudget(2), seed=8).optimize(
            objective, worker_pool=pool
        )
        first_pid = first.trials[0].observations[0].worker_pid
        second = hpo.Study([{"x": 0}], scheduler=hpo.FullBudget(2), seed=9).optimize(
            objective, worker_pool=pool
        )
        checked = hpo.audit(
            StatefulObjective(100, tmp_path / "audit"),
            {"x": 0},
            replications=2,
            worker_pool=pool,
        )
        assert first.trials[0].losses == second.trials[0].losses == [11, 12]
        assert [o.loss for o in checked.observations] == [101, 102]
        assert objective.calls == 0
        assert {o.worker_pid for o in checked.observations} == {first_pid}
        assert {o.worker_pid for o in second.trials[0].observations} == {first_pid}
        assert pool.worker_pids == (first_pid,)
        assert first.metadata["execution_backend"] == "persistent_processes"
        assert first.metadata["worker_pool_id"] == second.metadata["worker_pool_id"]
        assert first.metadata["worker_pool_run_id"] != second.metadata["worker_pool_run_id"]
        assert first.metadata["attempts"][0]["id"] == second.metadata["attempts"][0]["id"] == 1
        assert first.metadata["worker_pool_capacity"] == 1
        assert first.metadata["worker_pool_pids"] == [first_pid]
        assert (tmp_path / "audit" / "audit-1").exists()
        assert not (tmp_path / "first" / "audit-1").exists()
    assert pool.closed
    assert pool.worker_pids == ()
    pool.close()
    assert_stopped()


def test_pool_drains_attempt_allowance_and_resumes_without_changing_manifest(tmp_path):
    study = hpo.Study(
        [{"x": 0, "delay": 0.03}, {"x": 10, "delay": 0.03}],
        scheduler=hpo.FullBudget(3),
        workers=2,
        seed=7,
        storage=tmp_path / "study.sqlite",
        objective_id="pool-value-v1",
    )
    with pool_type()(workers=2) as pool:
        first = study.optimize(value, worker_pool=pool, budget=hpo.RunBudget(max_attempts=3))
        pids = set(pool.worker_pids)
        assert len(pids) == 2
        assert first.metadata["termination_reason"] == "attempt_budget"
        assert first.metadata["session_attempts"] == 3
        assert sum(len(t.observations) for t in first.trials) == 3
        second = study.optimize(value, worker_pool=pool)
        assert second.metadata["session_attempts"] == 3
        assert set(pool.worker_pids) == pids
        assert [t.losses for t in second.trials] == [[0, 1, 2], [10, 11, 12]]
        assert second.best_trial.id == 0
    assert study.optimize(value).metadata["session_attempts"] == 0
    assert_stopped()


def test_smaller_study_uses_only_its_requested_concurrency(tmp_path):
    with pool_type()(workers=3) as pool:
        hpo.Study(
            [{"x": x, "delay": 0.08} for x in range(3)],
            workers=3,
            scheduler=hpo.FullBudget(1),
        ).optimize(value, worker_pool=pool)
        assert len(pool.worker_pids) == 3
        result = hpo.Study(
            [{"x": x, "delay": 0.08} for x in range(4)],
            workers=1,
            scheduler=hpo.FullBudget(1),
        ).optimize(value, worker_pool=pool)
        observations = [t.observations[0] for t in result.trials]
        assert len({o.worker_pid for o in observations}) == 1
        assert all(
            a.finished_at <= b.started_at
            for a, b in zip(observations, observations[1:], strict=False)
        )
        assert len(pool.worker_pids) == 3
    assert_stopped()


def test_crash_retries_same_seed_and_pool_can_run_again(tmp_path):
    with pool_type()() as pool:
        result = hpo.Study(
            [{"x": 3, "marker": str(tmp_path / "crash")}],
            scheduler=hpo.FullBudget(2),
        ).optimize(crash_once, worker_pool=pool)
        assert result.best_trial.losses == [3, 4]
        attempts = result.metadata["attempts"]
        assert [a["status"] for a in attempts] == ["infrastructure_failed", "accepted", "accepted"]
        assert attempts[0]["replication"] == attempts[1]["replication"]
        assert (
            hpo.Study([{"x": 8}], scheduler=hpo.FullBudget(1))
            .optimize(value, worker_pool=pool)
            .best_trial.mean
            == 8
        )
    assert_stopped()


def test_evaluation_timeout_replaces_only_its_worker():
    with pool_type()(workers=2) as pool:
        result = hpo.Study(
            [{"x": 0, "block": True}, {"x": 1}],
            workers=2,
            scheduler=hpo.FullBudget(2),
        ).optimize(blocked, worker_pool=pool, budget=hpo.RunBudget(evaluation_timeout_seconds=0.5))
        assert result.trials[0].status == "failed"
        assert result.best_trial.losses == [1, 2]
        healthy_pid = result.best_trial.observations[0].worker_pid
        assert healthy_pid in pool.worker_pids
        assert result.metadata["attempts"][0]["status"] == "timed_out"
        assert (
            hpo.Study([{"x": 5}], scheduler=hpo.FullBudget(1))
            .optimize(value, worker_pool=pool)
            .best_trial.mean
            == 5
        )
    assert_stopped()


@pytest.mark.parametrize("stop", ["wall", "caller"])
def test_stopping_session_reaps_workers_and_pool_is_reusable(tmp_path, stop):
    marker = tmp_path / "started"
    token = hpo.CancellationToken()

    def cancel():
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        token.cancel()

    sender = threading.Thread(target=cancel) if stop == "caller" else None
    with pool_type()() as pool:
        if sender:
            sender.start()
        started = time.perf_counter()
        result = hpo.Study(
            [{"x": 2, "block": True, "marker": str(marker)}],
            scheduler=hpo.FullBudget(1),
            max_retries=0,
            storage=tmp_path / "study.sqlite",
            objective_id="pool-stop-v1",
        ).optimize(
            blocked,
            worker_pool=pool,
            budget=hpo.RunBudget(max_wall_seconds=0.6),
            cancellation=token if sender else None,
        )
        if sender:
            sender.join(timeout=1)
        assert time.perf_counter() - started < 5
        assert marker.exists()
        assert result.metadata["termination_reason"] == ("cancelled" if sender else "wall_budget")
        assert result.metadata["attempts"][0]["status"] == "cancelled"
        assert pool.worker_pids == ()
        assert not pool.closed
        assert (
            hpo.Study([{"x": 5}], scheduler=hpo.FullBudget(1))
            .optimize(value, worker_pool=pool)
            .best_trial.mean
            == 5
        )
    assert_stopped()


def test_exhausted_crash_retries_leave_empty_reusable_pool():
    with pool_type()() as pool:
        with pytest.raises(RuntimeError, match="retry budget"):
            hpo.Study([{"x": 0}], max_retries=0).optimize(crash, worker_pool=pool)
        assert pool.worker_pids == ()
        assert (
            hpo.Study([{"x": 4}], scheduler=hpo.FullBudget(1))
            .optimize(value, worker_pool=pool)
            .best_trial.mean
            == 4
        )
    assert_stopped()


def test_idle_dead_worker_is_retired_before_next_admission():
    with pool_type()() as pool:
        first = hpo.Study([{"x": 1}], scheduler=hpo.FullBudget(1)).optimize(value, worker_pool=pool)
        pid = first.best_trial.observations[0].worker_pid
        process = next(p for p in multiprocessing.active_children() if p.pid == pid)
        os.kill(pid, signal.SIGTERM)
        # Reap is owned by the pool. Waiting on its Process avoids a PID race.
        process.join(timeout=2)
        second = hpo.Study([{"x": 2}], scheduler=hpo.FullBudget(1), max_retries=0).optimize(
            value, worker_pool=pool, budget=hpo.RunBudget(max_attempts=1)
        )
        assert second.best_trial.mean == 2
        assert second.best_trial.observations[0].worker_pid != pid
        assert second.metadata["session_attempts"] == 1
    assert_stopped()


@pytest.mark.parametrize("settings", [{"workers": 0}, {"workers": True}, {"threads_per_worker": 0}])
def test_invalid_pool_settings(settings):
    with pytest.raises(ValueError):
        pool_type()(**settings)


@pytest.mark.parametrize("invalid", ["capacity", "threads", "closed", "type", "foreign"])
def test_invalid_pool_rejected_before_creating_storage(tmp_path, invalid, monkeypatch):
    pool = pool_type()()
    try:
        if invalid == "closed":
            pool.close()
        if invalid == "foreign":
            monkeypatch.setattr("aster_hpo.workers.os.getpid", lambda: -1)
        study = hpo.Study(
            [{"x": 0}],
            workers=2 if invalid == "capacity" else 1,
            threads_per_worker=2 if invalid == "threads" else 1,
            storage=tmp_path / "forbidden.sqlite",
            objective_id="pool-validation-v1",
        )
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            study.optimize(value, worker_pool=object() if invalid == "type" else pool)
        assert not (tmp_path / "forbidden.sqlite").exists()
        assert not (tmp_path / "forbidden.sqlite.lock").exists()
    finally:
        monkeypatch.undo()
        pool.close()


def test_pool_rejects_pickling_and_setting_changes():
    with pool_type()() as pool:
        with pytest.raises(TypeError, match="pickle"):
            pickle.dumps(pool)
        with pytest.raises(AttributeError):
            pool.workers = 2
        with pytest.raises(AttributeError):
            pool.threads_per_worker = 2


def test_concurrent_run_and_close_are_rejected_before_storage(tmp_path):
    marker = tmp_path / "started"
    results = queue.Queue()
    token = hpo.CancellationToken()
    pool = pool_type()()

    def run():
        try:
            results.put(
                hpo.Study([{"x": 0, "block": True, "marker": str(marker)}]).optimize(
                    blocked, worker_pool=pool, cancellation=token
                )
            )
        except BaseException as exc:
            results.put(exc)

    sender = threading.Thread(target=run)
    sender.start()
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        with pytest.raises(RuntimeError, match="use|borrow|running"):
            pool.close()
        with pytest.raises(RuntimeError, match="use|borrow|running"):
            hpo.Study(
                [{"x": 1}], storage=tmp_path / "forbidden.sqlite", objective_id="v1"
            ).optimize(value, worker_pool=pool)
        assert not (tmp_path / "forbidden.sqlite").exists()
    finally:
        token.cancel()
        sender.join(timeout=5)
        pool.close()
    assert not sender.is_alive()
    assert not isinstance(results.get_nowait(), BaseException)
    assert_stopped()


def test_child_thread_limits_and_parent_environment_are_preserved(monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("threadpoolctl")
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    with pool_type()(threads_per_worker=2) as pool:
        for _ in range(2):
            result = hpo.Study(
                [{"x": 0}], scheduler=hpo.FullBudget(1), threads_per_worker=2
            ).optimize(thread_setting, worker_pool=pool)
            assert result.best_trial.mean == 2
            assert os.environ["OMP_NUM_THREADS"] == "7"
    assert_stopped()


def test_pre_cancelled_call_reaps_existing_idle_pool_workers():
    with pool_type()() as pool:
        hpo.Study([{"x": 1}], scheduler=hpo.FullBudget(1)).optimize(value, worker_pool=pool)
        assert pool.worker_pids
        token = hpo.CancellationToken()
        token.cancel()
        result = hpo.Study([{"x": 2}], scheduler=hpo.FullBudget(1)).optimize(
            value, worker_pool=pool, cancellation=token
        )
        assert result.metadata["session_attempts"] == 0
        assert result.metadata["termination_reason"] == "cancelled"
        assert pool.worker_pids == ()
    assert_stopped()


def test_coordinator_accept_error_reaps_pool_and_preserves_recoverable_journal(
    tmp_path, monkeypatch
):
    from aster_hpo.runtime import Coordinator

    study = hpo.Study(
        [{"x": 3}],
        scheduler=hpo.FullBudget(1),
        storage=tmp_path / "study.sqlite",
        objective_id="accept-error-v1",
    )

    def fail_accept(*args):
        raise RuntimeError("fixture acceptance failure")

    with pool_type()() as pool:
        with monkeypatch.context() as patch:
            patch.setattr(Coordinator, "accept", fail_accept)
            with pytest.raises(RuntimeError, match="fixture acceptance"):
                study.optimize(value, worker_pool=pool)
        assert pool.worker_pids == ()
        resumed = study.optimize(value, worker_pool=pool)
        assert resumed.best_trial.mean == 3
        assert [a["status"] for a in resumed.metadata["attempts"]] == ["interrupted", "accepted"]
    assert_stopped()


def test_prior_run_response_cannot_satisfy_reused_attempt_id(tmp_path, monkeypatch):
    from aster_hpo.workers import Worker

    with pool_type()() as pool:
        first = hpo.Study([{"x": 0}], scheduler=hpo.FullBudget(1)).optimize(
            StatefulObjective(10, tmp_path / "first"), worker_pool=pool
        )
        stale = first.best_trial.observations[0]
        submit = Worker.submit

        def submit_with_stale_response(worker, job, dispatched, **kwargs):
            # A delayed old response has an otherwise valid replication and
            # attempt ID. Only its independently generated run token differs.
            worker.responses.put(
                ("result", first.metadata["worker_pool_run_id"], 1, stale, time.perf_counter())
            )
            submit(worker, job, dispatched, **kwargs)

        monkeypatch.setattr(Worker, "submit", submit_with_stale_response)
        second = hpo.Study([{"x": 0}], scheduler=hpo.FullBudget(1)).optimize(
            StatefulObjective(100, tmp_path / "second"), worker_pool=pool
        )
        assert second.best_trial.mean == 101
        assert second.metadata["session_attempts"] == 1
        assert (tmp_path / "second" / "search-0").exists()
    assert_stopped()


def test_unpicklable_binding_reaps_pool_and_next_run_can_succeed():
    with pool_type()() as pool:
        with pytest.raises((AttributeError, TypeError), match="pickle"):
            hpo.Study([{"x": 0}], scheduler=hpo.FullBudget(1)).optimize(
                lambda c, r: 0, worker_pool=pool
            )
        assert pool.worker_pids == ()
        assert (
            hpo.Study([{"x": 2}], scheduler=hpo.FullBudget(1))
            .optimize(value, worker_pool=pool)
            .best_trial.mean
            == 2
        )
    assert_stopped()


def test_storage_error_reaps_previously_idle_pool_workers(tmp_path):
    with pool_type()() as pool:
        settings = {
            "storage": tmp_path / "study.sqlite",
            "objective_id": "manifest-v1",
            "scheduler": hpo.FullBudget(1),
        }
        hpo.Study([{"x": 1}], **settings).optimize(value, worker_pool=pool)
        assert pool.worker_pids
        with pytest.raises(ValueError, match="manifest"):
            hpo.Study([{"x": 2}], **settings).optimize(value, worker_pool=pool)
        assert pool.worker_pids == ()
        assert (
            hpo.Study([{"x": 3}], scheduler=hpo.FullBudget(1))
            .optimize(value, worker_pool=pool)
            .best_trial.mean
            == 3
        )
    assert_stopped()


def test_concurrent_close_does_not_close_process_handles_twice(monkeypatch):
    from aster_hpo import workers

    pool = pool_type()()
    hpo.Study([{"x": 1}], scheduler=hpo.FullBudget(1)).optimize(value, worker_pool=pool)
    entered, finish = threading.Event(), threading.Event()
    errors = queue.Queue()
    stop_workers = workers.stop_workers

    def delayed_stop(group, *, force):
        owned = list(group)
        if not entered.is_set():
            entered.set()
            assert finish.wait(5)
        stop_workers(owned, force=force)

    def close():
        try:
            pool.close()
        except BaseException as exc:
            errors.put(exc)

    monkeypatch.setattr(workers, "stop_workers", delayed_stop)
    first = threading.Thread(target=close)
    second = threading.Thread(target=close)
    first.start()
    assert entered.wait(2)
    second.start()
    second.join(timeout=0.1)
    finish.set()
    first.join(timeout=3)
    second.join(timeout=3)
    assert not first.is_alive() and not second.is_alive()
    assert list(errors.queue) == []
    assert pool.closed
    assert_stopped()
