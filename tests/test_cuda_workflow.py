"""Integrity and small real-workflow checks for the optional CUDA comparison."""

import copy
import json
import math
import subprocess
import sys

import pytest


def _protocol():
    from benchmarks.cuda_workflow import default_protocol

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
    protocol["seeds"] = [31000]
    protocol["count"] = 2
    protocol["checkpoints"] = [2, 3]
    protocol["audit_replications"] = 2
    return protocol


def _job():
    return {
        "id": "tiny-31000-numpy-full-w1",
        "workload": "tiny",
        "seed": 31000,
        "backend": "numpy",
        "scheduler": "full",
        "workers": 1,
    }


def _run_campaign(tmp_path):
    from benchmarks.cuda_workflow import source_identity

    protocol = _protocol()
    source = source_identity()
    folder = tmp_path / "campaign"
    request = tmp_path / "job.json"
    request.write_text(
        json.dumps({"job": _job(), "protocol": protocol, "source": source, "folder": str(folder)})
    )
    completed = subprocess.run(
        [sys.executable, "-m", "benchmarks.cuda_workflow", "campaign", "--job", str(request)],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads((folder / "record.json").read_text())
    record["timings"] = json.loads((folder / "timing.json").read_text())
    return folder, protocol, record


def test_frozen_protocol_has_thirty_paired_interleaved_jobs_and_cuda_header():
    from benchmarks.cuda_workflow import build_jobs, default_protocol, source_identity

    protocol = default_protocol()
    assert protocol["seeds"] == [31000, 31001, 31002]
    assert protocol["count"] == 4
    assert protocol["checkpoints"] == [2, 4]
    assert protocol["audit_replications"] == 3
    assert protocol["markets"]["tiny"]["batch_paths"] == 128
    assert protocol["markets"]["tiny"]["evaluation_paths"] == 2048
    assert protocol["markets"]["scaled"]["batch_paths"] == 4096
    assert protocol["markets"]["scaled"]["evaluation_paths"] == 32768
    assert all(m["epochs"] == 64 and m["steps"] == 12 for m in protocol["markets"].values())
    jobs = build_jobs(protocol)
    assert len(jobs) == len({job["id"] for job in jobs}) == 30
    assert jobs == build_jobs(protocol)
    assert {job["backend"] for job in jobs[:10]} == {"numpy", "cuda"}
    assert {job["workload"] for job in jobs[:10]} == {"tiny", "scaled"}
    assert sum(job["workers"] == 4 for job in jobs) == 6
    assert all(
        job["scheduler"] == "full" and job["backend"] == "numpy"
        for job in jobs
        if job["workers"] == 4
    )
    source = source_identity()
    assert source["sha256"]
    assert "cuda/include/accounting_kernel.cuh" in source["files"]


def test_existing_output_is_preserved_before_launching_jobs(tmp_path):
    from benchmarks.cuda_workflow import run

    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        run(output)
    assert marker.read_text() == "keep"
    assert list(output.iterdir()) == [marker]


def test_real_campaign_saves_journals_winner_independent_metrics_and_timing(tmp_path):
    from benchmarks.cuda_workflow import validate_record

    folder, protocol, record = _run_campaign(tmp_path)
    validate_record(record, protocol)
    assert (folder / "search.sqlite").is_file()
    assert (folder / "audit.sqlite").is_file()
    assert json.loads((folder / "winner.json").read_text()) == record["winner"]
    assert record["counts"] == {
        "search_observations": 6,
        "audit_observations": 2,
        "objective_failures": 0,
        "unsuccessful_attempts": 0,
    }
    assert len(record["metrics"]) == 8
    assert all((folder / row["file"]).is_file() for row in record["metrics"])
    search = [
        o["replication"]["scenario_seed"]
        for t in record["search"]["trials"]
        for o in t["observations"]
    ]
    audits = [o["replication"]["scenario_seed"] for o in record["audit"]["observations"]]
    assert set(search).isdisjoint(audits)
    times = record["timings"]
    assert times["workflow_seconds"] >= sum(
        times[k]
        for k in (
            "backend_setup_seconds",
            "search_seconds",
            "audit_seconds",
            "result_write_seconds",
        )
    )
    assert times["host_cpu_seconds"] >= 0
    assert record["campaign_pid"] != __import__("os").getpid()
    assert record["preloaded_numerical_modules"] == []
    assert record["timing_scope"]["warmup_objective_calls"] == 0
    assert "result writes" in record["timing_scope"]["workflow"]
    assert "host" in record["timing_scope"]["cpu"]


@pytest.mark.parametrize("corruption", ["failed", "missing", "nonfinite", "identity", "count"])
def test_invalid_campaign_cannot_hide_behind_a_completed_winner(tmp_path, corruption):
    from benchmarks.cuda_workflow import validate_record

    _, protocol, record = _run_campaign(tmp_path)
    observation = record["search"]["trials"][1]["observations"][0]
    if corruption == "failed":
        observation["loss"] = None
        observation["error"] = "injected failure"
    elif corruption == "missing":
        record["search"]["trials"][1]["observations"].pop()
    elif corruption == "nonfinite":
        observation["loss"] = math.nan
    elif corruption == "identity":
        observation["replication"]["scenario_seed"] += 1
    else:
        record["counts"]["search_observations"] -= 1
    with pytest.raises(ValueError):
        validate_record(record, protocol)


def test_missing_metric_file_and_failed_attempt_are_rejected(tmp_path):
    from benchmarks.cuda_workflow import validate_record

    folder, protocol, record = _run_campaign(tmp_path)
    damaged = copy.deepcopy(record)
    damaged["search"]["metadata"]["attempts"][0]["status"] = "infrastructure_failed"
    with pytest.raises(ValueError, match="attempt"):
        validate_record(damaged, protocol)
    (folder / record["metrics"][0]["file"]).unlink()
    with pytest.raises(ValueError, match="metric"):
        validate_record(record, protocol, folder=folder)


def test_summary_pairs_seeds_before_ratios_and_checks_every_matched_loss(tmp_path):
    from benchmarks.cuda_workflow import summarize, validate_summary

    _, protocol, original = _run_campaign(tmp_path)
    records = []
    for seed, cpu_time, gpu_time in [(31000, 10.0, 2.0), (31001, 12.0, 4.0)]:
        for backend, wall in [("numpy", cpu_time), ("cuda", gpu_time)]:
            row = copy.deepcopy(original)
            # Test summary arithmetic with already validated records. Search identities
            # remain tied to the real original run and are identical within each pair.
            row["job"].update(seed=seed, backend=backend, id=f"{seed}-{backend}")
            row["timings"]["workflow_seconds"] = wall
            row["process_seconds"] = wall + 1.0
            if seed == 31001 and backend == "cuda":
                row["search"]["trials"][1]["observations"][2]["loss"] += 0.01
            records.append(row)
    summary = summarize(records, protocol)
    comparison = summary["full_budget"][0]
    assert comparison["workflow_speedup"]["median"] == 4.0
    assert comparison["workflow_speedup"]["minimum"] == 3.0
    assert comparison["workflow_speedup"]["maximum"] == 5.0
    assert comparison["matched_losses"]["count"] == 12
    assert comparison["matched_losses"]["mismatches"] == 1
    assert comparison["matched_losses"]["all_close"] is False
    assert comparison["winner_agreement_count"] == 2
    assert all(row["audit_mean_difference"] == 0 for row in comparison["pairs"])
    assert not summary["adaptive"]
    with pytest.raises(ValueError, match="closeness"):
        validate_summary(summary)


def test_summary_refuses_unpaired_or_duplicate_campaigns(tmp_path):
    from benchmarks.cuda_workflow import summarize

    _, protocol, row = _run_campaign(tmp_path)
    with pytest.raises(ValueError, match="pair"):
        summarize([row], protocol)
    duplicate = copy.deepcopy(row)
    with pytest.raises(ValueError, match="duplicate"):
        summarize([row, duplicate], protocol)


def test_counts_must_be_integer_evidence(tmp_path):
    from benchmarks.cuda_workflow import validate_record

    _, protocol, record = _run_campaign(tmp_path)
    record["counts"]["search_observations"] = 6.0
    with pytest.raises(ValueError, match="count"):
        validate_record(record, protocol)


def test_real_objective_failure_precludes_lock_even_when_study_has_a_winner(tmp_path, monkeypatch):
    from benchmarks.cuda_workflow import campaign, source_identity
    from benchmarks.hedging import HedgingObjective, hedging_pool

    protocol = _protocol()
    bad = hedging_pool(31000, protocol["count"])[1]
    original = HedgingObjective.__call__

    def fail_one(self, config, replication):
        if config == bad:
            raise RuntimeError("injected objective failure")
        return original(self, config, replication)

    monkeypatch.setattr(HedgingObjective, "__call__", fail_one)
    folder = tmp_path / "failed-campaign"
    with pytest.raises(ValueError, match="search trial"):
        campaign(
            {
                "job": _job(),
                "protocol": protocol,
                "source": source_identity(),
                "folder": str(folder),
            }
        )
    failure = json.loads((folder / "failure.json").read_text())
    assert any(o["error"] for o in failure["search"]["observations"])
    search = json.loads((folder / "search.json").read_text())
    assert search["best_trial_id"] is not None
    assert not (folder / "winner.json").exists()
    assert not (folder / "record.json").exists()


@pytest.mark.parametrize("field", ["protocol", "backend", "metric_backend"])
def test_record_metadata_cannot_change_backend_or_protocol(tmp_path, field):
    from benchmarks.cuda_workflow import validate_record

    _, protocol, record = _run_campaign(tmp_path)
    if field == "protocol":
        record["protocol_sha256"] = "corrupt"
    elif field == "backend":
        record["backend"]["name"] = "cuda"
    else:
        record["metrics"][0]["backend"] = "cuda"
    with pytest.raises(ValueError):
        validate_record(record, protocol)


def test_adaptive_comparison_keeps_different_work_counts_descriptive(tmp_path):
    from benchmarks.cuda_workflow import summarize, validate_summary

    _, protocol, cpu = _run_campaign(tmp_path)
    gpu = copy.deepcopy(cpu)
    cpu["job"]["scheduler"] = gpu["job"]["scheduler"] = "race"
    gpu["job"]["backend"] = "cuda"
    gpu["job"]["id"] = "gpu-race"
    gpu["search"]["trials"][1]["observations"].pop()
    gpu["search"]["trials"][0]["observations"][0]["loss"] += 0.01
    summary = summarize([cpu, gpu], protocol)
    validate_summary(summary)
    assert not summary["full_budget"]
    pair = summary["adaptive"][0]["pairs"][0]
    assert pair["numpy_observations"] == 6
    assert pair["cuda_observations"] == 5
    assert pair["numpy_unmatched_observations"] == 1
    assert pair["cuda_unmatched_observations"] == 0
    assert not summary["adaptive"][0]["matched_losses"]["all_close"]


def test_runner_freezes_manifest_and_starts_a_fresh_process_per_job(tmp_path, monkeypatch):
    import benchmarks.cuda_workflow as workflow

    protocol = _protocol()
    one = _job()
    four = {**one, "workers": 4, "id": one["id"].replace("w1", "w4")}
    monkeypatch.setattr(workflow, "default_protocol", lambda: protocol)
    monkeypatch.setattr(workflow, "build_jobs", lambda _: [one, four])
    # This test isolates orchestration; numerical paired summaries are covered above.
    monkeypatch.setattr(
        workflow,
        "summarize",
        lambda *_: {"full_budget": [], "adaptive": [], "cpu_four_worker_control": []},
    )
    output = tmp_path / "run"
    workflow.run(output)
    manifest = json.loads((output / "manifest.json").read_text())
    records = json.loads((output / "records.json").read_text())
    assert manifest["jobs"] == [one, four]
    assert len({r["campaign_pid"] for r in records}) == 2
    assert all(r["process_seconds"] >= r["timings"]["workflow_seconds"] for r in records)
    assert all(r["preloaded_numerical_modules"] == [] for r in records)
    assert (
        len({o["worker_pid"] for t in records[1]["search"]["trials"] for o in t["observations"]})
        > 1
    )
    assert all(
        r["backend"]["environment"]["thread_environment"]["OPENBLAS_NUM_THREADS"] == "1"
        for r in records
    )
    assert all(r["source_sha256"] == manifest["source"]["sha256"] for r in records)


def test_accepted_attempt_must_match_the_observation_identity(tmp_path):
    from benchmarks.cuda_workflow import validate_record

    _, protocol, record = _run_campaign(tmp_path)
    record["search"]["metadata"]["attempts"][0]["replication"]["training_seed"] += 1
    with pytest.raises(ValueError, match="attempt"):
        validate_record(record, protocol)
