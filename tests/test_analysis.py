import math

import pytest

pytest.importorskip("scipy")


def test_one_sided_bound_matches_hand_computation():
    from benchmarks.analysis import paired_analysis

    result = paired_analysis([1, 2, 3], margin=5)
    assert result["mean_difference"] == 2
    assert result["standard_error"] == pytest.approx(1 / math.sqrt(3))
    assert result["upper_bound"] == pytest.approx(3.685854461, rel=1e-7)
    assert result["status"] == "noninferior"
    assert paired_analysis([1, 2, 3], margin=1)["status"] == "inconclusive"


def test_missing_or_insufficient_runs_cannot_support_preservation():
    from benchmarks.analysis import paired_analysis

    assert paired_analysis([None, 0, 0], margin=1)["status"] == "incomplete"
    assert paired_analysis([0], margin=1)["status"] == "insufficient"
    with pytest.raises(ValueError):
        paired_analysis([1, float("nan")], margin=1)
    with pytest.raises(ValueError):
        paired_analysis([1, 2], margin=-1)


@pytest.mark.parametrize("difference", [0.0, 0.01, -0.01])
def test_constant_paired_differences_have_no_inferential_bound(difference):
    from benchmarks.analysis import paired_analysis

    result = paired_analysis([difference] * 20, margin=1)
    assert result["mean_difference"] == difference
    assert result["standard_error"] == 0
    assert result["upper_bound"] is None
    assert result["status"] == "degenerate"


def test_plot_interval_requires_multiple_independent_runs():
    from benchmarks.study_plots import mean_interval

    assert mean_interval([1]) is None
    assert mean_interval([1, 2, 3]) == pytest.approx(2.484137712, rel=1e-7)


def test_nested_replications_reduce_to_one_difference_per_optimizer_seed():
    from benchmarks.analysis import analyze_campaigns

    records = [
        {
            "seed": seed,
            "method": method,
            "audit_losses": audit_losses,
            "search_wall_seconds": seconds,
            "objective_cpu_seconds": seconds,
            "replications": 10,
            "audit_wall_seconds": 1,
            "audit_cpu_seconds": 1,
            "allocated_worker_seconds": seconds,
            "unknown_attempts": 0,
        }
        for seed, baseline, challenger in [(0, [1, 3], [2, 4]), (1, [10, 10], [9, 9])]
        for method, audit_losses, seconds in [
            ("parallel_full", baseline, 3),
            ("race", challenger, 2),
        ]
    ]
    report = analyze_campaigns(records, margin=0.1)
    comparison = report["comparisons"]["race"]
    assert comparison["differences"] == [1, -1]
    assert comparison["n"] == 2
    assert comparison["mean_difference"] == 0
    assert report["methods"]["race"]["median_speedup"] == 1.5
