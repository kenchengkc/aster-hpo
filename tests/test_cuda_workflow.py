"""Integrity and small real-workflow checks for the optional CUDA comparison."""

import copy
import json
import math
import subprocess
import sys
from dataclasses import asdict

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
    monkeypatch.setattr(workflow, "default_protocol", lambda **_: protocol)
    monkeypatch.setattr(workflow, "build_jobs", lambda _: [one, four])
    # This test isolates orchestration; numerical paired summaries are covered above.
    monkeypatch.setattr(
        workflow,
        "summarize",
        lambda *_: {"full_budget": [], "adaptive": [], "cpu_four_worker_control": []},
    )
    output = tmp_path / "run"
    execute = workflow.subprocess.run

    def frozen_requests(*args, **kwargs):
        # Every request is immutable evidence before the first campaign clock starts.
        if args[0][1:3] == ["-m", "benchmarks.cuda_workflow"]:
            assert len(list((output / "jobs").glob("*.json"))) == 2
        return execute(*args, **kwargs)

    monkeypatch.setattr(workflow.subprocess, "run", frozen_requests)
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


def test_eager_comparison_is_opt_in_and_adds_twelve_fresh_campaigns():
    from benchmarks.cuda_workflow import build_jobs, default_protocol

    original = build_jobs(default_protocol())
    protocol = default_protocol(include_eager=True)
    jobs = build_jobs(protocol)
    assert len(jobs) == len({job["id"] for job in jobs}) == 42
    assert jobs == build_jobs(protocol)
    eager = [job for job in jobs if job["backend"] == "cuda_eager"]
    assert len(eager) == 12
    assert {(job["workload"], job["scheduler"], job["seed"]) for job in eager} == {
        (workload, scheduler, seed)
        for workload in ("tiny", "scaled")
        for scheduler in ("full", "race")
        for seed in (31000, 31001, 31002)
    }
    assert all(job["workers"] == 1 for job in eager)
    assert {job["id"] for job in original} == {
        job["id"] for job in jobs if job["backend"] != "cuda_eager"
    }
    assert {job["backend"] for job in jobs[:14]} == {"numpy", "cuda", "cuda_eager"}
    assert {job["workload"] for job in jobs[:14]} == {"tiny", "scaled"}
    assert protocol["warmup_objective_calls"] == 0


@pytest.mark.parametrize("include_eager", [False, True])
def test_cli_selects_optional_eager_campaigns(tmp_path, monkeypatch, include_eager):
    import benchmarks.cuda_workflow as workflow

    output = tmp_path / "output"

    def collect(path, *, progress, include_eager=False):
        path.mkdir()
        (path / "selected.json").write_text(json.dumps({"eager": include_eager}))

    monkeypatch.setattr(workflow, "run", collect)
    args = ["run", "--output", str(output)]
    if include_eager:
        args.append("--include-eager")
    workflow.main(args)
    assert json.loads((output / "selected.json").read_text()) == {"eager": include_eager}


def _optimization_records(original, timings=((31000, 10.0, 2.0),)):
    records = []
    for seed, eager_time, optimized_time in timings:
        for backend, wall in (("cuda_eager", eager_time), ("cuda", optimized_time)):
            row = copy.deepcopy(original)
            row["job"].update(seed=seed, backend=backend, id=f"{seed}-{backend}")
            row["timings"]["workflow_seconds"] = wall
            row["process_seconds"] = wall + 1.0
            records.append(row)
    return records


def test_optimization_ratios_are_paired_and_implementation_names_are_accurate(tmp_path):
    from benchmarks.cuda_workflow import summarize, validate_summary

    _, protocol, original = _run_campaign(tmp_path)
    records = _optimization_records(original, ((31000, 10.0, 2.0), (31001, 12.0, 4.0)))
    summary = validate_summary(summarize(records, protocol))
    group = summary["cuda_optimization"][0]
    assert group["workflow_speedup"] == {"median": 4.0, "minimum": 3.0, "maximum": 5.0}
    assert group["process_speedup"]["median"] == pytest.approx((11 / 3 + 13 / 5) / 2)
    assert group["eager_workflow_seconds"] == {"median": 11.0, "minimum": 10.0, "maximum": 12.0}
    assert group["optimized_process_seconds"] == {"median": 4.0, "minimum": 3.0, "maximum": 5.0}
    assert group["matched_losses"]["count"] == 12
    assert group["matched_metrics"]["count"] == 112
    assert group["matched_metrics"]["all_close"]
    assert group["allocation_agreement_count"] == 2
    assert group["winner_agreement_count"] == 2
    assert group["pairs"][0]["eager_job"] == "31000-cuda_eager"
    assert group["pairs"][0]["optimized_job"] == "31000-cuda"
    assert all(not key.startswith("numpy") for key in group)
    assert all(not key.startswith("numpy") for row in group["pairs"] for key in row)
    assert not summary["full_budget"]


@pytest.mark.parametrize("workers,category", [(1, "full_budget"), (4, "cpu_four_worker_control")])
@pytest.mark.parametrize(
    "metric",
    ["bias", "error_q05", "error_q50", "error_q95", "mean_transaction_cost", "mean_turnover"],
)
def test_eager_protocol_cpu_controls_require_every_matched_financial_metric(
    tmp_path, workers, category, metric
):
    from benchmarks.cuda_workflow import summarize, validate_summary

    _, protocol, cpu = _run_campaign(tmp_path)
    protocol["include_eager"] = True
    cpu["job"]["workers"] = workers
    eager, optimized = _optimization_records(cpu)
    eager["job"]["workers"] = optimized["job"]["workers"] = 1
    cpu["metrics"][0]["metrics"][metric] += 0.01
    summary = summarize([cpu, eager, optimized], protocol)
    with pytest.raises(ValueError, match="closeness"):
        validate_summary(summary)
    assert summary[category][0]["matched_metrics"]["mismatches"] == 1


def test_eager_protocol_speedup_definitions_name_the_actual_implementations(tmp_path):
    from benchmarks.cuda_workflow import summarize

    _, protocol, cpu = _run_campaign(tmp_path)
    protocol["include_eager"] = True
    eager, optimized = _optimization_records(cpu)
    summary = summarize([cpu, eager, optimized], protocol)
    assert summary["full_budget"][0]["speedup_definition"] == "numpy_seconds / cuda_seconds"
    assert summary["cuda_optimization"][0]["speedup_definition"] == (
        "eager_seconds / optimized_seconds"
    )


def _near_tied_optimization_records(original):
    from aster_hpo import Replication

    records = _optimization_records(original)
    for winner_id, record in enumerate(records):
        backend = record["job"]["backend"]
        record["backend"].update(
            name=backend, implementation="eager" if winner_id == 0 else "optimized"
        )
        record["winner"] = record["search"]["trials"][winner_id]["config"]
        record["winner_trial_id"] = record["search"]["best_trial_id"] = winner_id
        losses = {}
        for trial in record["search"]["trials"]:
            loss = 1.0 - (5e-9 if winner_id == 1 and trial["id"] == 1 else 0.0)
            losses[json.dumps(trial["config"], sort_keys=True)] = loss
            for observation in trial["observations"]:
                observation["loss"] = loss
        for observation in record["audit"]["observations"]:
            index = observation["replication"]["index"]
            observation["replication"] = asdict(
                Replication.create(record["job"]["seed"], record["winner"], index, "audit", True)
            )
            observation["loss"] = 3.0 + winner_id
        for attempt in record["audit"]["metadata"]["attempts"]:
            attempt["replication"] = record["audit"]["observations"][attempt["replication_index"]][
                "replication"
            ]
        for metric in record["metrics"]:
            metric["backend"] = backend
            if metric["replication"]["phase"] == "search":
                metric["metrics"]["mse"] = losses[json.dumps(metric["config"], sort_keys=True)]
            else:
                index = metric["replication"]["index"]
                metric["config"] = record["winner"]
                metric["replication"] = record["audit"]["observations"][index]["replication"]
                metric["metrics"]["mse"] = 3.0 + winner_id
                replication = metric["replication"]
                metric["file"] = (
                    f"metrics/audit-{replication['training_seed']:016x}-"
                    f"{replication['scenario_seed']:016x}-{index}.json"
                )
    return records


def test_close_searches_can_lock_different_winners_and_report_unmatched_audits(tmp_path):
    from benchmarks.cuda_workflow import summarize, validate_record, validate_summary

    _, protocol, original = _run_campaign(tmp_path)
    records = _near_tied_optimization_records(original)
    for record in records:
        validate_record(record, protocol)
    summary = validate_summary(summarize(records, protocol))
    group = summary["cuda_optimization"][0]
    assert group["matched_losses"]["all_close"]
    assert group["matched_metrics"]["count"] == 42
    assert group["winner_agreement_count"] == 0
    pair = group["pairs"][0]
    assert not pair["winner_agreement"]
    assert pair["eager_unmatched_metric_observations"] == 2
    assert pair["optimized_unmatched_metric_observations"] == 2
    assert pair["audit_mean_difference"] == 1.0


@pytest.mark.parametrize("phase", ["search", "audit"])
@pytest.mark.parametrize("both", [False, True])
def test_fixed_budget_metric_evidence_cannot_be_omitted(tmp_path, phase, both):
    from benchmarks.cuda_workflow import summarize

    _, protocol, original = _run_campaign(tmp_path)
    eager, optimized = _optimization_records(original)
    for record in (eager, optimized) if both else (optimized,):
        index = next(
            i for i, row in enumerate(record["metrics"]) if row["replication"]["phase"] == phase
        )
        record["metrics"].pop(index)
    with pytest.raises(ValueError, match="metric"):
        summarize([eager, optimized], protocol)


@pytest.mark.parametrize("phase", ["search", "audit"])
@pytest.mark.parametrize(
    "metric",
    [
        "mse",
        "bias",
        "error_q05",
        "error_q50",
        "error_q95",
        "mean_transaction_cost",
        "mean_turnover",
    ],
)
def test_optimization_fixed_budget_rejects_each_mismatched_financial_metric(
    tmp_path, phase, metric
):
    from benchmarks.cuda_workflow import summarize, validate_summary

    _, protocol, original = _run_campaign(tmp_path)
    eager, optimized = _optimization_records(original)
    row = next(row for row in optimized["metrics"] if row["replication"]["phase"] == phase)
    row["metrics"][metric] += 0.01
    summary = summarize([eager, optimized], protocol)
    assert summary["cuda_optimization"][0]["matched_metrics"]["mismatches"] == 1
    with pytest.raises(ValueError, match="closeness"):
        validate_summary(summary)


def test_optimization_fixed_budget_rejects_different_search_identities(tmp_path):
    from benchmarks.cuda_workflow import summarize

    _, protocol, original = _run_campaign(tmp_path)
    eager, optimized = _optimization_records(original)
    optimized["search"]["trials"][1]["observations"].pop()
    with pytest.raises(ValueError, match="identit"):
        summarize([eager, optimized], protocol)


def test_eager_protocol_refuses_missing_eager_campaigns(tmp_path):
    from benchmarks.cuda_workflow import summarize

    _, protocol, cpu = _run_campaign(tmp_path)
    protocol["include_eager"] = True
    optimized = _optimization_records(cpu)[1]
    with pytest.raises(ValueError, match="paired eager"):
        summarize([cpu, optimized], protocol)


def test_optimization_adaptive_allocation_differences_stay_descriptive(tmp_path):
    from benchmarks.cuda_workflow import summarize, validate_summary

    _, protocol, original = _run_campaign(tmp_path)
    eager, optimized = _optimization_records(original)
    eager["job"]["scheduler"] = optimized["job"]["scheduler"] = "race"
    optimized["search"]["trials"][1]["observations"].pop()
    optimized["metrics"].pop(5)
    summary = validate_summary(summarize([eager, optimized], protocol))
    group = summary["cuda_optimization"][0]
    assert group["allocation_agreement_count"] == 0
    assert group["pairs"][0]["eager_observations"] == 6
    assert group["pairs"][0]["optimized_observations"] == 5
    assert group["pairs"][0]["eager_unmatched_observations"] == 1
    assert group["matched_metrics"]["all_close"]


def test_optimization_adaptive_matched_numerics_must_still_be_close(tmp_path):
    from benchmarks.cuda_workflow import summarize, validate_summary

    _, protocol, original = _run_campaign(tmp_path)
    eager, optimized = _optimization_records(original)
    eager["job"]["scheduler"] = optimized["job"]["scheduler"] = "race"
    optimized["metrics"][0]["metrics"]["bias"] += 0.01
    summary = summarize([eager, optimized], protocol)
    assert not summary["cuda_optimization"][0]["matched_metrics"]["all_close"]
    with pytest.raises(ValueError, match="closeness"):
        validate_summary(summary)


@pytest.mark.parametrize("backend,implementation", [("cuda", "optimized"), ("cuda_eager", "eager")])
def test_cuda_campaign_records_shared_device_and_distinct_implementation(
    tmp_path, monkeypatch, backend, implementation
):
    from benchmarks import hedging_cuda
    from benchmarks.cuda_workflow import campaign, source_identity, validate_record
    from benchmarks.hedging import HedgingObjective

    original_evaluate = HedgingObjective.evaluate

    def cpu_evaluate(self, config, replication):
        return original_evaluate(HedgingObjective(self.market), config, replication)

    monkeypatch.setattr(HedgingObjective, "evaluate", cpu_evaluate)
    monkeypatch.setattr(hedging_cuda, "device_info", lambda: {"device_name": "test device"})
    monkeypatch.setattr(hedging_cuda, "synchronize", lambda: None)
    protocol = _protocol()
    record = campaign(
        {
            "job": {**_job(), "backend": backend},
            "protocol": protocol,
            "source": source_identity(),
            "folder": str(tmp_path / backend),
        }
    )
    assert record["backend"]["device"] == {"device_name": "test device"}
    assert record["backend"]["implementation"] == implementation
    assert all(row["backend"] == backend for row in record["metrics"])
    record["backend"]["implementation"] = "wrong"
    with pytest.raises(ValueError, match="implementation"):
        validate_record(record, protocol)


def test_search_validation_can_fix_candidates_while_replication_seeds_change(tmp_path):
    from aster_hpo import FullBudget, Study
    from benchmarks.cuda_workflow import _read, _validate_search
    from benchmarks.hedging import HedgingObjective, Market, hedging_pool

    market = Market(epochs=1, batch_paths=4, evaluation_paths=16)
    result = Study(hedging_pool(71, 2), scheduler=FullBudget(2), seed=72).optimize(
        HedgingObjective(market)
    )
    result.save(tmp_path / "search.json")
    winner, count = _validate_search(
        _read(tmp_path / "search.json"),
        {"seed": 72, "pool_seed": 71, "scheduler": "full"},
        {"count": 2, "checkpoints": [2]},
    )
    assert count == 4 and winner["config"] == result.best_trial.config
