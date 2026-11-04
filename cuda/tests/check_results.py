"""Fail a remote run if recorded evidence lacks completed, checked work."""

import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for name, mode in [("smoke", "smoke"), ("correctness", "correctness"), ("benchmark", "benchmark")]:
    with (root / (name + ".json")).open() as stream:
        result = json.load(stream)
    assert result["mode"] == mode
    assert result["passed"] is True
    assert result["gpu_owning_processes_in_runner"] == 1
    assert result["comparison"]["atol"] == 1e-12
    assert result["comparison"]["rtol"] == 1e-10
    assert result["comparison"]["outputs"] == ["pnl", "costs", "turnover"]
    assert result["comparison"]["values_checked"] > 0
    for field in ("max_absolute_error", "max_normalized_error"):
        assert math.isfinite(result["comparison"][field]) and result["comparison"][field] >= 0
    assert result["comparison"]["max_normalized_error"] <= 1
    assert result["device"]["name"]
    assert result["device"]["total_memory_bytes"] > 0
    assert result["device"]["driver_version"] > 0
    assert result["device"]["runtime_version"] > 0
    assert "--fmad=false" in result["build"]["cuda_flags"]
    assert result["cold"]["runtime_setup_ms"] >= 0
    assert result["cold"]["device_allocation_ms"] >= 0
    assert result["cold"]["first_completed_host_ms"] > 0
    if mode == "correctness":
        required = {
            "resident_lifecycle_invalid_launch_rate_slot_incomplete_and_consumed_outputs",
            "resident_same_shape_changed_input_reload",
            "resident_changed_shape_reload",
        }
        for fixture in (
            "hand_checked",
            "short_single_interval",
            "empty_paths",
            "selected_block_partial",
            "exact_cancelling_gain",
            "near_cancelling_gain",
            "gain_minus_cost_cancellation",
            "random_257x256_0.010000",
        ):
            required.update(f"resident_{fixture}_operations_{operations}" for operations in (1, 3))
        assert required <= set(result["cases"]), sorted(required - set(result["cases"]))
    if mode == "benchmark":
        samples = result["samples"]
        assert len(samples) == result["workload"]["repeats"] > 0
        for sample in samples:
            for field in ("kernel_event_ms", "completed_host_ms", "native_cpu_prepared_ms"):
                assert math.isfinite(sample[field]) and sample[field] > 0, (field, sample)
        assert all(math.isfinite(v) for v in result["checksum"].values())
        assert result["workload"]["input_hash_fnv1a64"]
print("Result checks passed: correctness metadata and completed positive timing samples.")
