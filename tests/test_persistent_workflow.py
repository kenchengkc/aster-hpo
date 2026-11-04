"""Integrity checks for complete repeated workflows and observed process reuse."""

import copy
import json
import os
import subprocess
import sys

import pytest


def _protocol():
    from benchmarks.persistent_workflow import default_protocol

    protocol = default_protocol()
    protocol["markets"] = {
        "tiny": {
            **protocol["markets"]["tiny"],
            "epochs": 1,
            "steps": 2,
            "batch_paths": 4,
            "evaluation_paths": 8,
        }
    }
    protocol["sequence_roots"] = [41000]
    protocol["count"] = 2
    protocol["checkpoints"] = [1]
    protocol["audit_replications"] = 2
    return protocol


def _run_sequence(tmp_path, mode="direct", workers=1, *, root=41000):
    from benchmarks.persistent_workflow import build_jobs, source_identity

    protocol = _protocol()
    protocol["sequence_roots"] = [root]
    job = next(
        row
        for row in build_jobs(protocol)
        if row["backend"] == "numpy" and row["mode"] == mode and row["workers"] == workers
    )
    folder = tmp_path / f"{root}-{mode}-{workers}"
    request = tmp_path / f"request-{root}-{mode}-{workers}.json"
    request.write_text(
        json.dumps(
            {"job": job, "protocol": protocol, "source": source_identity(), "folder": str(folder)}
        )
    )
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.persistent_workflow", "sequence", "--job", str(request)],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads((folder / "sequence.json").read_text())
    record["timings"] = json.loads((folder / "sequence-timing.json").read_text())
    return folder, protocol, record


@pytest.fixture(scope="module")
def direct_sequence(tmp_path_factory):
    return _run_sequence(tmp_path_factory.mktemp("repeated-direct"))


def test_fixed_jobs_cover_144_new_workflows_and_2736_evaluations():
    from benchmarks.persistent_workflow import build_jobs, default_protocol

    protocol = default_protocol()
    jobs = build_jobs(protocol)
    assert len(jobs) == len({row["id"] for row in jobs}) == 48
    assert jobs == build_jobs(protocol)
    assert protocol["sequence_roots"] == [41000, 42000, 43000]
    assert protocol["workflow_offsets"] == [0, 1, 2]
    assert protocol["count"] == 4 and protocol["checkpoints"] == [4]
    assert protocol["audit_replications"] == 3
    assert len(jobs) * len(protocol["workflow_offsets"]) == 144
    assert 144 * (4 * 4 + 3) == 2736
    assert {(row["mode"], row["backend"], row["workers"]) for row in jobs} == {
        ("direct", "numpy", 1),
        ("direct", "cuda", 1),
        ("fresh_pool", "numpy", 1),
        ("fresh_pool", "cuda", 1),
        ("fresh_pool", "numpy", 4),
        ("reused_pool", "numpy", 1),
        ("reused_pool", "cuda", 1),
        ("reused_pool", "numpy", 4),
    }
    assert protocol["markets"]["tiny"]["batch_paths"] == 128
    assert protocol["markets"]["tiny"]["evaluation_paths"] == 2048
    assert protocol["markets"]["scaled"]["batch_paths"] == 4096
    assert protocol["markets"]["scaled"]["evaluation_paths"] == 32768
    assert all(m["epochs"] == 64 and m["steps"] == 12 for m in protocol["markets"].values())


@pytest.mark.parametrize("kind", ["directory", "empty", "symlink"])
def test_existing_destination_is_preserved_before_imports_or_subprocess(tmp_path, kind):
    from benchmarks.persistent_workflow import run

    output = tmp_path / "existing"
    if kind == "symlink":
        output.symlink_to(tmp_path / "absent", target_is_directory=True)
    else:
        output.mkdir()
        if kind == "directory":
            (output / "marker").write_text("keep")
    with pytest.raises(FileExistsError):
        run(output)
    assert output.is_symlink() if kind == "symlink" else output.is_dir()


def test_real_direct_sequence_keeps_complete_fresh_workflows_and_boundaries(direct_sequence):
    from benchmarks.persistent_workflow import validate_sequence

    folder, protocol, record = direct_sequence
    validate_sequence(record, protocol, folder=folder)
    assert record["preloaded_numerical_modules"] == []
    assert record["coordinator_pid"] != os.getpid()
    assert [run["job"]["seed"] for run in record["runs"]] == [41000, 41001, 41002]
    for index, run in enumerate(record["runs"]):
        path = folder / f"workflow-{index}"
        assert (path / "search.sqlite").is_file() and (path / "audit.sqlite").is_file()
        assert json.loads((path / "winner.json").read_text()) == run["winner"]
        assert run["counts"]["search_observations"] == 2
        assert run["counts"]["audit_observations"] == 2
        assert len(run["metrics"]) == 4
        assert len(run["worker_identities"]) == 1
        identity = run["worker_identities"][0]
        assert identity["pid"] == record["coordinator_pid"]
        assert identity["backend"]["name"] == "numpy"
        assert "device" not in identity["backend"]
        assert "environment" not in identity
        assert identity["numerical_modules"] == ["numpy"]
        assert set(identity["thread_environment"].values()) == {"1"}
        assert run["timings"]["workflow_seconds"] >= sum(
            run["timings"][part]
            for part in (
                "backend_setup_seconds",
                "search_seconds",
                "audit_seconds",
                "result_write_seconds",
                "pool_close_seconds",
            )
        )
    times = [run["timings"]["workflow_seconds"] for run in record["runs"]]
    assert record["timings"]["first_workflow_seconds"] == times[0]
    assert record["timings"]["subsequent_workflow_seconds"] == times[1:]
    assert record["timings"]["subsequent_median_seconds"] == sum(times[1:]) / 2
    assert record["timings"]["whole_sequence_seconds"] >= sum(times)
    assert record["timings"]["amortized_workflow_seconds"] == (
        record["timings"]["whole_sequence_seconds"] / 3
    )
    assert record["reuse"]["adjacent_reused_processes"] == [1, 1]
    assert record["cleanup"]["surviving_worker_pids"] == []


@pytest.mark.parametrize(
    "mode,workers", [("fresh_pool", 1), ("reused_pool", 1), ("reused_pool", 4)]
)
def test_real_numpy_pool_sequence_proves_rebinding_reuse_and_cleanup(tmp_path, mode, workers):
    from benchmarks.persistent_workflow import validate_sequence

    folder, protocol, record = _run_sequence(tmp_path, mode, workers)
    validate_sequence(record, protocol, folder=folder)
    assert all(
        run["search"]["metadata"]["execution_backend"] == "persistent_processes"
        for run in record["runs"]
    )
    assert all(
        row["pid"] != record["coordinator_pid"]
        for run in record["runs"]
        for row in run["worker_identities"]
    )
    if mode == "fresh_pool":
        assert record["reuse"]["adjacent_reused_processes"] == [0, 0]
        assert all(run["timings"]["pool_close_seconds"] > 0 for run in record["runs"])
    else:
        assert all(count >= 1 for count in record["reuse"]["adjacent_reused_processes"])
        assert record["timings"]["final_pool_close_seconds"] > 0
    assert record["cleanup"]["surviving_worker_pids"] == []
    for pid in record["cleanup"]["managed_worker_pids"]:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize(
    "damage", ["partial", "attempt", "metric", "pid", "nonce", "reuse", "counts"]
)
def test_corrupt_sequence_evidence_is_rejected(direct_sequence, damage):
    from benchmarks.persistent_workflow import validate_sequence

    _, protocol, original = direct_sequence
    record = copy.deepcopy(original)
    if damage == "partial":
        record["runs"].pop()
    elif damage == "attempt":
        record["runs"][0]["audit"]["metadata"]["attempts"].pop()
    elif damage == "metric":
        record["runs"][0]["metrics"].pop()
    elif damage == "pid":
        record["runs"][0]["worker_identities"][0]["pid"] += 1
    elif damage == "nonce":
        record["runs"][1]["worker_identities"][0]["process_nonce"] = ""
    elif damage == "reuse":
        record["reuse"]["adjacent_reused_processes"] = [0, 0]
    else:
        record["runs"][0]["counts"]["search_observations"] -= 1
    with pytest.raises(ValueError):
        validate_sequence(record, protocol)


def test_pool_metadata_must_match_pid_evidence_capacity_and_independent_leases(tmp_path):
    from benchmarks.persistent_workflow import validate_sequence

    _, protocol, original = _run_sequence(tmp_path, "reused_pool")
    for field, value in [
        ("worker_pool_pids", []),
        ("worker_pool_capacity", 4),
        ("worker_pool_id", "another-pool"),
        ("worker_pool_run_id", original["runs"][0]["search"]["metadata"]["worker_pool_run_id"]),
    ]:
        record = copy.deepcopy(original)
        record["runs"][1]["audit"]["metadata"][field] = value
        with pytest.raises(ValueError, match="pool|lease|PID"):
            validate_sequence(record, protocol)


@pytest.mark.parametrize("mode,workers", [("direct", 1), ("fresh_pool", 1), ("reused_pool", 4)])
def test_accepted_observation_pids_match_each_phase_and_worker_evidence(tmp_path, mode, workers):
    from benchmarks.persistent_workflow import validate_sequence

    folder, protocol, original = _run_sequence(tmp_path, mode, workers)
    validate_sequence(original, protocol, folder=folder)
    for phase in ("search", "audit"):
        for all_observations in (False, True):
            record = copy.deepcopy(original)
            run = record["runs"][0]
            observations = (
                [obs for trial in run["search"]["trials"] for obs in trial["observations"]]
                if phase == "search"
                else run["audit"]["observations"]
            )
            for observation in observations if all_observations else observations[:1]:
                observation["worker_pid"] = -999
            with pytest.raises(ValueError, match="PID"):
                validate_sequence(record, protocol, folder=folder)
        for invalid_pid in (None, True, 1.5):
            record = copy.deepcopy(original)
            observation = (
                record["runs"][0]["search"]["trials"][0]["observations"][0]
                if phase == "search"
                else record["runs"][0]["audit"]["observations"][0]
            )
            observation["worker_pid"] = invalid_pid
            with pytest.raises(ValueError, match="PID"):
                validate_sequence(record, protocol, folder=folder)
    record = copy.deepcopy(original)
    run = record["runs"][0]
    if mode == "direct":
        # A valid positive integer must still identify the evaluating coordinator.
        run["audit"]["observations"][0]["worker_pid"] += 1
    else:
        audit_pids = set(run["audit"]["metadata"]["worker_pool_pids"])
        search_pids = set(run["search"]["metadata"]["worker_pool_pids"])
        if workers == 4:
            # Search still accounts for the union: only the audit/phase link is broken.
            replacement = next(iter(search_pids - audit_pids))
            for observation in run["audit"]["observations"]:
                observation["worker_pid"] = replacement
        else:
            run["audit"]["observations"][0]["worker_pid"] = record["coordinator_pid"]
    with pytest.raises(ValueError, match="PID"):
        validate_sequence(record, protocol, folder=folder)


def test_direct_observations_require_the_in_process_backend(direct_sequence):
    from benchmarks.persistent_workflow import validate_sequence

    folder, protocol, original = direct_sequence
    record = copy.deepcopy(original)
    record["runs"][0]["audit"]["metadata"]["execution_backend"] = "persistent_processes"
    with pytest.raises(ValueError, match="backend"):
        validate_sequence(record, protocol, folder=folder)


def test_summary_uses_one_subsequent_median_per_independent_sequence(direct_sequence):
    from benchmarks.cuda_workflow import _digest
    from benchmarks.persistent_workflow import summarize, validate_summary

    _, protocol, original = direct_sequence
    protocol = copy.deepcopy(protocol)
    protocol["variants"] = [
        {"mode": "direct", "backend": "numpy", "workers": 1},
        {"mode": "direct", "backend": "cuda", "workers": 1},
    ]
    records = []
    for backend, durations, total in [("numpy", [10, 8, 4], 24), ("cuda", [5, 1, 3], 12)]:
        row = copy.deepcopy(original)
        row["protocol_sha256"] = _digest(protocol)
        row["job"].update(id=f"tiny-41000-direct-{backend}-w1", backend=backend)
        for run, duration in zip(row["runs"], durations, strict=True):
            run["protocol_sha256"] = row["protocol_sha256"]
            run["job"]["backend"] = backend
            run["backend"] = {
                "name": backend,
                **({"implementation": "optimized"} if backend == "cuda" else {}),
            }
            for identity in run["worker_identities"]:
                identity["backend"] = {
                    **run["backend"],
                    **({"device": {"name": "fixture"}} if backend == "cuda" else {}),
                }
            for metric in run["metrics"]:
                metric["backend"] = backend
            run["timings"].update(workflow_seconds=duration)
        row["timings"].update(
            first_workflow_seconds=durations[0],
            subsequent_workflow_seconds=durations[1:],
            subsequent_median_seconds=sum(durations[1:]) / 2,
            whole_sequence_seconds=total,
            amortized_workflow_seconds=total / 3,
        )
        row["process_seconds"] = total + 1
        records.append(row)
    summary = summarize(records, protocol)
    pair = summary["comparisons"][0]
    assert pair["independent_sequences"] == 1
    assert pair["first_workflow_speedup"]["median"] == 2
    assert pair["subsequent_median_speedup"]["median"] == 3
    assert pair["amortized_workflow_speedup"]["median"] == 2
    assert pair["matched_metrics"]["count"] == 84
    validate_summary(summary)
    records[1]["runs"][0]["metrics"][0]["metrics"]["bias"] += 0.01
    damaged = summarize(records, protocol)
    with pytest.raises(ValueError, match="closeness"):
        validate_summary(damaged)
    with pytest.raises(ValueError, match="Missing"):
        summarize(records[:1], protocol)
    with pytest.raises(ValueError, match="duplicate"):
        summarize([*records, records[0]], protocol)


def test_summary_pairs_reused_cpu_and_cuda_with_equal_persistence(tmp_path, direct_sequence):
    from benchmarks.cuda_workflow import _digest
    from benchmarks.persistent_workflow import summarize, validate_summary

    _, protocol, first = direct_sequence
    protocol = copy.deepcopy(protocol)
    protocol["sequence_roots"] = [41000, 42000, 43000]
    protocol["variants"] = [
        {"mode": "direct", "backend": "numpy", "workers": 1},
        {"mode": "reused_pool", "backend": "numpy", "workers": 1},
        {"mode": "reused_pool", "backend": "numpy", "workers": 4},
        {"mode": "reused_pool", "backend": "cuda", "workers": 1},
    ]
    # Retain real seed/configuration/metric evidence for all three outer roots.
    originals = [first, *[_run_sequence(tmp_path, root=root)[2] for root in (42000, 43000)]]
    durations = [
        [[40, 16, 8], [40, 16, 8], [40, 16, 8]],
        [[20, 8, 4], [24, 20, 10], [30, 2, 4]],
        [[10, 4, 2], [12, 10, 5], [15, 1, 2]],
        [[5, 1, 3], [6, 2, 4], [10, 2, 4]],
    ]
    totals = [[72, 120, 120], [36, 60, 60], [18, 30, 30], [12, 20, 20]]
    records = []
    for outer, original in enumerate(originals):
        for index, variant in enumerate(protocol["variants"]):
            row = copy.deepcopy(original)
            root = row["job"]["root_seed"]
            name = f"{variant['mode']}-{variant['backend']}-w{variant['workers']}"
            row["protocol_sha256"] = _digest(protocol)
            row["job"].update(**variant, id=f"tiny-{root}-{name}")
            worker_pid = row["coordinator_pid"]
            pooled = variant["mode"] == "reused_pool"
            if pooled:
                row["coordinator_pid"] += 10000 + index
                row["cleanup"]["managed_worker_pids"] = [worker_pid]
            for position, run in enumerate(row["runs"]):
                run["protocol_sha256"] = row["protocol_sha256"]
                run["job"].update(**variant, id=f"{row['job']['id']}-workflow-{position}")
                run["campaign_pid"] = row["coordinator_pid"]
                run["backend"] = {"name": variant["backend"]}
                if variant["backend"] == "cuda":
                    run["backend"]["implementation"] = "optimized"
                for identity in run["worker_identities"]:
                    identity["backend"] = copy.deepcopy(run["backend"])
                    if variant["backend"] == "cuda":
                        identity["backend"]["device"] = {"name": "arithmetic fixture"}
                        identity["numerical_modules"] = ["numpy", "cupy", "benchmarks.hedging_cuda"]
                for phase in ("search", "audit"):
                    if pooled:
                        run[phase]["metadata"].update(
                            execution_backend="persistent_processes",
                            workers=variant["workers"] if phase == "search" else 1,
                            worker_pool_id=f"fixture-{root}-{name}",
                            worker_pool_run_id=f"fixture-{root}-{name}-{position}-{phase}",
                            worker_pool_capacity=variant["workers"],
                            worker_pool_pids=[worker_pid],
                        )
                for metric in run["metrics"]:
                    metric["backend"] = variant["backend"]
                run["timings"]["workflow_seconds"] = durations[index][outer][position]
            total = totals[index][outer]
            row["timings"].update(
                first_workflow_seconds=durations[index][outer][0],
                subsequent_workflow_seconds=durations[index][outer][1:],
                subsequent_median_seconds=sum(durations[index][outer][1:]) / 2,
                final_pool_close_seconds=0.25 if pooled else 0,
                whole_sequence_seconds=total,
                amortized_workflow_seconds=total / 3,
            )
            row["process_seconds"] = total + 1
            records.append(row)
    summary = summarize(records, protocol)
    validate_summary(summary)
    pairs = {(group["reference"], group["candidate"]): group for group in summary["comparisons"]}
    for workers, expected, first_ratio, amortized in [
        (1, [1, 3, 5], 4, 3),
        (4, [0.5, 1.5, 2.5], 2, 1.5),
    ]:
        reference = f"reused_pool-numpy-w{workers}"
        group = pairs[reference, "reused_pool-cuda-w1"]
        assert group["reference"] == reference
        assert group["independent_sequences"] == 3
        assert [pair["root_seed"] for pair in group["pairs"]] == [41000, 42000, 43000]
        assert group["subsequent_median_speedup"] == {
            "minimum": expected[0],
            "median": expected[1],
            "maximum": expected[2],
        }
        assert group["first_workflow_speedup"]["median"] == first_ratio
        assert group["amortized_workflow_speedup"]["median"] == amortized
        assert group["matched_metrics"]["count"] == 252
    assert ("direct-numpy-w1", "reused_pool-cuda-w1") in pairs
    records[-1]["runs"][0]["metrics"][0]["metrics"]["bias"] += 0.01
    with pytest.raises(ValueError, match="closeness"):
        validate_summary(summarize(records, protocol))
