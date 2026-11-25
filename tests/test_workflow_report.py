"""Regressions for offline reporting of the archived A100 campaign timings."""

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.cuda_workflow import _read
from benchmarks.workflow_report import EVIDENCE, render_report


@pytest.fixture(scope="module")
def evidence():
    return _read(EVIDENCE / "manifest.json"), _read(EVIDENCE / "records.json")


def test_archived_medians_and_paired_speedups_without_site_packages():
    run = subprocess.run(
        [sys.executable, "-S", "-m", "benchmarks.workflow_report"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [" ".join(line.split()) for line in run.stdout.splitlines()]
    # The small eager workflow's ratio of medians would be 5.01, not 4.955.
    assert "128 / 2,048 Eager CUDA 4.625 4.955× 4.929 4.182×" in lines
    assert "128 / 2,048 Optimized CUDA 0.923 1.000× 1.165 1.000×" in lines
    assert "4,096 / 32,768 NumPy (1 worker) 16.720 4.639× 16.809 4.339×" in lines
    assert "4,096 / 32,768 NumPy (4 workers) 8.865 2.565× 8.970 2.414×" in lines


def test_entire_missing_seed_and_duplicate_campaign_rejected(evidence):
    manifest, records = evidence
    with pytest.raises(ValueError, match=r"Missing full-budget campaigns.*31000"):
        render_report(manifest, [r for r in records if r["job"]["seed"] != 31000])
    with pytest.raises(ValueError, match=r"[Dd]uplicate.*tiny-31000-numpy-full-w1"):
        render_report(manifest, [*records, records[0]])


@pytest.mark.parametrize("duration", [float("inf"), 0, True])
def test_invalid_duration_names_campaign_and_clock(evidence, duration):
    manifest, records = evidence
    bad = {**records[0], "process_seconds": duration}
    with pytest.raises(ValueError, match="tiny-31000-numpy-full-w1.*process_seconds"):
        render_report(manifest, [bad, *records[1:]])
