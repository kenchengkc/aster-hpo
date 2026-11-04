"""Reject incorrect residency evidence before deriving performance summaries."""

import copy
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "benchmark_pipeline.py"


def collector():
    spec = importlib.util.spec_from_file_location("aster_pipeline_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_record():
    methods = ["copy_per_call", "resident_session", "resident_preloaded", "native_cpu_sequence"]
    sample = {
        "method_order": methods,
        "copy_per_call_ms": 8.0,
        "resident_session_ms": 2.0,
        "resident_preloaded_ms": 1.0,
        "native_cpu_sequence_ms": 4.0,
        "transfers": {
            "copy_per_call": {"h2d_bytes": 96, "d2h_bytes": 96, "h2d_calls": 4, "d2h_calls": 6},
            "resident_session": {"h2d_bytes": 48, "d2h_bytes": 96, "h2d_calls": 2, "d2h_calls": 3},
            "resident_preloaded": {"h2d_bytes": 0, "d2h_bytes": 96, "h2d_calls": 0, "d2h_calls": 3},
        },
    }
    return {
        "schema_version": 2,
        "passed": True,
        "mode": "pipeline",
        "device": {"name": "test device", "compute_capability": "8.0"},
        "build": {"float_type": "float64", "cuda_flags": "--fmad=false"},
        "workload": {
            "paths": 2,
            "intervals": 1,
            "operations": 2,
            "rates": [0.0, 0.01],
            "repeats": 2,
            "input_hash_fnv1a64": "0123456789abcdef",
        },
        "comparison": {
            "passed": True,
            "all_operations_checked": True,
            "values_checked": 120,
            "outputs": ["pnl", "costs", "turnover"],
            "atol": 1e-12,
            "rtol": 1e-10,
            "max_normalized_error": 0.1,
            "max_absolute_error": 1e-14,
        },
        "checksum": {
            method: {
                "pnl": 1.0,
                "costs": 2.0,
                "turnover": 3.0,
                "ordered_hash_fnv1a64": "0011223344556677",
            }
            for method in methods
        },
        "pipeline_samples": [copy.deepcopy(sample), copy.deepcopy(sample)],
    }


def test_complete_session_summary_uses_process_medians():
    module = collector()
    record = valid_record()
    module.validate_record(record, paths=2, intervals=1, operations=2, repeats=2)
    first = {"session": 0, "record": record}
    other = copy.deepcopy(record)
    other["pipeline_samples"][0]["copy_per_call_ms"] = 4.0
    other["pipeline_samples"][1]["copy_per_call_ms"] = 4.0
    summary = module.summarize_records([first, {"session": 1, "record": other}])
    assert len(summary) == 1
    row = summary[0]
    assert row["process_sessions"] == 2
    assert row["copy_per_call_median_ms"] == 6.0
    assert row["resident_session_median_ms"] == 2.0
    assert row["paired_session_speedup_median"] == 3.0
    assert row["paired_session_speedup_range"] == [2.0, 4.0]


@pytest.mark.parametrize(
    "fault",
    [
        "failed",
        "nan",
        "missing_output",
        "copies",
        "readback",
        "preloaded_upload",
        "order",
        "rate",
        "checks",
        "samples",
        "checksum",
        "comparison_failed",
        "partial_operations",
    ],
)
def test_invalid_or_incomparable_records_are_rejected(fault):
    record = valid_record()
    sample = record["pipeline_samples"][0]
    if fault == "failed":
        record["passed"] = False
    elif fault == "nan":
        sample["resident_session_ms"] = float("nan")
    elif fault == "missing_output":
        record["comparison"]["outputs"] = ["pnl"]
    elif fault == "copies":
        sample["transfers"]["resident_session"]["h2d_calls"] = 4
    elif fault == "readback":
        sample["transfers"]["resident_session"]["d2h_bytes"] = 48
    elif fault == "preloaded_upload":
        sample["transfers"]["resident_preloaded"]["h2d_bytes"] = 48
    elif fault == "order":
        sample["method_order"][-1] = "resident_session"
    elif fault == "rate":
        record["workload"]["rates"] = [0.01, 0.01]
    elif fault == "checks":
        record["comparison"]["values_checked"] = 12
    elif fault == "samples":
        record["pipeline_samples"].pop()
    elif fault == "checksum":
        record["checksum"]["resident_session"]["ordered_hash_fnv1a64"] = "ffffffffffffffff"
    elif fault == "comparison_failed":
        record["comparison"]["passed"] = False
    else:
        record["comparison"]["all_operations_checked"] = False
    with pytest.raises(ValueError):
        collector().validate_record(record, paths=2, intervals=1, operations=2, repeats=2)


def test_existing_evidence_is_preserved_before_binary_execution(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker.json"
    marker.write_text(json.dumps({"keep": True}))
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--binary", "/missing/binary", "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "already exists" in completed.stderr
    assert json.loads(marker.read_text()) == {"keep": True}
