"""Small behavioral checks for the optional native measurement commands."""

import json
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy", reason="measurement tools require the benchmark extra")


def _strict_json(text):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


def test_api_benchmark_records_real_samples_and_preserves_existing_output(tmp_path):
    output = tmp_path / "measurement"
    command = [
        sys.executable,
        "-m",
        "benchmarks.native_performance",
        str(output),
        "--numpy-only",
        "--sizes",
        "1",
        "4",
        "--intervals",
        "2",
        "--samples",
        "2",
        "--warmup",
        "1",
        "--sessions",
        "1",
        "--cold-samples",
        "1",
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    report = _strict_json((output / "report.json").read_text())
    assert report["settings"]["sizes"] == [1, 4]
    assert report["settings"]["processes"] == 1
    assert report["provenance"]["native_source_sha256"]
    assert report["timer"]["samples_ns"]
    for row in report["api"]:
        assert row["backend"] == "numpy"
        assert len(row["samples_ns"]) == 2
        assert all(sample > 0 for sample in row["samples_ns"])
        assert np.isfinite(row["checksum"])
        assert row["workload_sha256"]
        assert (output / row["input_file"]).is_file()
    assert len(report["cold"]) == 2
    assert all(row["first_call_ns"] > 0 for row in report["cold"])
    assert all(row["process_wall_ns"] > 0 for row in report["cold"])
    original = (output / "report.json").read_bytes()
    collision = subprocess.run(command, text=True, capture_output=True)
    assert collision.returncode != 0
    assert (output / "report.json").read_bytes() == original


def test_wrong_accounting_is_rejected_before_reporting():
    from benchmarks.native_performance import compare_outputs

    expected = (np.array([0.089]), np.array([0.011]), np.array([1.1]))
    with pytest.raises(AssertionError, match="pnl"):
        compare_outputs(expected, (np.array([0.09]), expected[1], expected[2]))


def test_native_measurements_use_exact_accounting_fixture():
    native = pytest.importorskip("aster_hpo_native", reason="optional companion must be installed")
    from benchmarks.native_performance import compare_outputs

    prices = np.array([[1.0, 1.2]])
    holdings = np.array([[0.5]])
    expected = (np.array([0.089]), np.array([0.011]), np.array([1.1]))
    compare_outputs(expected, native.hedge_pnl(prices, holdings, cost_rate=0.01))
    outputs = tuple(np.empty(1) for _ in range(3))
    native.hedge_pnl_into(prices, holdings, *outputs, cost_rate=0.01)
    compare_outputs(expected, outputs)


def test_standalone_benchmark_json_and_collision_behavior(tmp_path):
    import os

    executable = os.environ.get("ASTER_ACCOUNTING_BENCH")
    if executable is None:
        pytest.skip("set ASTER_ACCOUNTING_BENCH to a built standalone executable")
    output = tmp_path / "standalone"
    command = [
        executable,
        "--output",
        str(output),
        "--paths",
        "4",
        "--intervals",
        "2",
        "--samples",
        "31",
        "--warmup",
        "1",
        "--events",
        "32",
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    report = _strict_json(completed.stdout)
    assert _strict_json((output / "report.json").read_text()) == report
    assert report["correctness"]["passed"]
    assert report["provenance"]["source_fnv1a64"]
    assert report["provenance"]["binary_fnv1a64"]
    for row in report["batch"]:
        assert len(row["samples_ns"]) == 31
        assert all(value >= 0 for value in row["samples_ns"])
        assert sum(row["samples_ns"]) > 0
        assert np.isfinite(row["checksum"])
    streaming = report["streaming"]
    assert len(streaming["update_samples_ns"]) == 32
    assert all(value >= 0 for value in streaming["update_samples_ns"])
    assert sum(streaming["update_samples_ns"]) > 0
    assert len(streaming["events"]) > 32  # Finish events retain their chronological position.
    assert streaming["update_quantiles_ns"]["p99"] > 0
    assert streaming["quantile_method"] == "linear interpolation at (n-1)*q"
    original = (output / "report.json").read_bytes()
    collision = subprocess.run(command, text=True, capture_output=True)
    assert collision.returncode != 0
    assert (output / "report.json").read_bytes() == original


@pytest.mark.parametrize(("paths", "intervals"), [(1, 1), (4, 2)])
def test_standalone_tiny_measurements_emit_finite_or_unavailable_rates(paths, intervals):
    import os

    executable = os.environ.get("ASTER_ACCOUNTING_BENCH")
    if executable is None:
        pytest.skip("set ASTER_ACCOUNTING_BENCH to a built standalone executable")
    completed = subprocess.run(
        [
            executable,
            "--paths",
            str(paths),
            "--intervals",
            str(intervals),
            "--samples",
            "101",
            "--warmup",
            "1",
            "--events",
            "32",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = _strict_json(completed.stdout)
    for row in report["batch"]:
        median = row["call_quantiles_ns"]["p50"]
        if median == 0:
            assert row["median_paths_per_second"] is None
            assert row["median_ns_per_path"] is None
        else:
            assert row["median_paths_per_second"] == pytest.approx(paths * 1e9 / median)
            assert row["median_ns_per_path"] == pytest.approx(median / paths)
