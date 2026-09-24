import json

import pytest

from aster_hpo import FullBudget, Study


def loss(config, replication):
    return config["x"] ** 2


def fails(config, replication):
    raise ValueError("fixture failure")


def test_regret_ties_and_all_failed_are_explicit():
    from benchmarks.diagnostics import summarize_result

    result = Study([{"x": -1}, {"x": 1}, {"x": 3}], scheduler=FullBudget(2)).optimize(loss)
    summary = summarize_result(result, "additive")
    assert summary["pool_regret"] == 0
    assert summary["continuous_regret"] == 1
    assert summary["optimum_count"] == 2
    result.trials[0].status = "pruned"
    summary = summarize_result(result, "additive")
    assert summary["any_optimum_pruned"]
    assert not summary["all_optima_pruned"]
    failed = Study([{"x": 0}], scheduler=FullBudget(2)).optimize(fails)
    summary = summarize_result(failed, "additive")
    assert summary["pool_regret"] is None
    assert summary["winner"] is None
    assert summary["failed_trials"] == 1


def test_suite_reproduces_serial_outcomes_and_rejects_collisions(tmp_path):
    from benchmarks.diagnostics import run_suite

    reports = []
    for directory in ("a", "b"):
        report = run_suite(tmp_path / directory, seeds=(7,), workers=1, count=3, checkpoints=(3, 9))
        assert report["interpretation"] == "synthetic diagnostic"
        assert report["runs"] == 32
        assert (tmp_path / directory / "report.md").is_file()
        reports.append(report)
    # This report contains statistical outcomes, not runtime measurements.
    assert reports[0] == reports[1]
    with pytest.raises(FileExistsError):
        run_suite(tmp_path / "a", seeds=(8,), workers=1, count=3, checkpoints=(3, 9))


def test_report_uses_artifacts_and_refuses_incomplete_data(tmp_path):
    from benchmarks.diagnostics import regenerate, run_suite

    report = run_suite(tmp_path / "run", seeds=(7,), count=3, checkpoints=(3, 9))
    assert regenerate(tmp_path / "run") == report
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["environment"]["revision"]
    assert manifest["environment"]["python"]
    first = tmp_path / "run" / "runs" / (manifest["runs"][0]["id"] + ".json")
    first.unlink()
    with pytest.raises(FileNotFoundError):
        regenerate(tmp_path / "run")


def test_paired_summary_preserves_adverse_results_and_separates_order():
    from benchmarks.diagnostics import aggregate

    records = [
        dict(
            case="near_ties",
            shared=True,
            order=order,
            seed=seed,
            method=method,
            pool_regret=regret,
            replications=reps,
            any_optimum_pruned=False,
        )
        for order in ("forward", "reverse")
        for seed in (0, 1)
        for method, regret, reps in (("full_budget", 0.0, 9), ("race", 0.5, 3))
    ]
    report = aggregate(records)
    assert len(report["paired"]) == 2
    assert all(row["differences"] == [0.5, 0.5] for row in report["paired"])
    assert all(row["independent_seeds"] == 2 for row in report["paired"])


def test_synthetic_noise_has_residual_uncertainty_and_finite_heavy_tails():
    from aster_hpo import Replication
    from benchmarks.synthetic import SyntheticObjective, expected_loss

    configs = [{"x": 0}, {"x": 1}]
    differences = []
    objective = SyntheticObjective("heteroscedastic")
    for i in range(10):
        observations = [
            objective(config, Replication.create(7, config, i, "search", True))
            for config in configs
        ]
        differences.append(observations[1] - observations[0])
    assert len(set(differences)) > 1
    assert expected_loss("near_ties", {"x": 2}) == pytest.approx(0.08)
    with pytest.raises(ValueError):
        SyntheticObjective("unknown")


def test_manifest_records_installed_native_thread_controller():
    import importlib.metadata

    from benchmarks.common import environment

    pytest.importorskip("threadpoolctl")
    assert environment()["dependencies"]["threadpoolctl"] == importlib.metadata.version(
        "threadpoolctl"
    )
