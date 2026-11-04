"""Collect completed, numerically checked accounting sessions from the CUDA runner."""

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

METHODS = ("copy_per_call", "resident_session", "resident_preloaded", "native_cpu_sequence")
GPU_METHODS = METHODS[:3]
ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, *, positive=False):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def validate_record(record, *, paths, intervals, operations, repeats):
    """Reject incorrect outputs, incomplete samples, and unequal transfer contracts."""
    try:
        require(
            record["schema_version"] == 2 and record["mode"] == "pipeline",
            "not a pipeline v2 record",
        )
        require(record["passed"] is True, "correctness did not pass")
        workload = record["workload"]
        for key, expected in (
            ("paths", paths),
            ("intervals", intervals),
            ("operations", operations),
            ("repeats", repeats),
        ):
            require(
                type(workload[key]) is int and workload[key] == expected, f"wrong workload {key}"
            )
        rates = workload["rates"]
        require(
            len(rates) == operations and all(finite(rate) and rate <= 0.01 for rate in rates),
            "invalid rates",
        )
        expected_rates = (
            [0.01]
            if operations == 1
            else [0.01 * index / (operations - 1) for index in range(operations)]
        )
        require(
            all(
                math.isclose(a, b, abs_tol=1e-15)
                for a, b in zip(rates, expected_rates, strict=True)
            ),
            "rates change the workload",
        )
        require(bool(workload["input_hash_fnv1a64"]), "missing input identity")
        require(
            record["device"]["name"] and record["build"]["float_type"] == "float64",
            "missing device or precision",
        )
        require(
            "--fmad=false" in record["build"]["cuda_flags"],
            "floating-point contraction is not disabled",
        )
        comparison = record["comparison"]
        require(
            comparison["passed"] is True and comparison["all_operations_checked"] is True,
            "not every operation passed comparison",
        )
        require(
            comparison["outputs"] == ["pnl", "costs", "turnover"], "not all outputs were compared"
        )
        require(
            comparison["atol"] == 1e-12 and comparison["rtol"] == 1e-10,
            "wrong numerical tolerances",
        )
        require(
            finite(comparison["max_normalized_error"]) and comparison["max_normalized_error"] <= 1,
            "numerical tolerance exceeded",
        )
        require(finite(comparison["max_absolute_error"]), "invalid numerical error")
        require(
            comparison["values_checked"] >= paths * operations * repeats * 9,
            "insufficient all-output comparisons",
        )
        samples = record["pipeline_samples"]
        require(len(samples) == repeats, "missing completed samples")
        input_bytes = paths * (2 * intervals + 1) * 8
        output_bytes = paths * operations * 3 * 8
        expected_copies = {
            "copy_per_call": (
                input_bytes * operations,
                output_bytes,
                operations * 2,
                operations * 3,
            ),
            "resident_session": (input_bytes, output_bytes, 2, 3),
            "resident_preloaded": (0, output_bytes, 0, 3),
        }
        for sample in samples:
            require(
                len(sample["method_order"]) == 4 and set(sample["method_order"]) == set(METHODS),
                "invalid method order",
            )
            for method in METHODS:
                require(
                    finite(sample[f"{method}_ms"], positive=method in GPU_METHODS),
                    f"invalid {method} duration",
                )
            for method, expected in expected_copies.items():
                copies = sample["transfers"][method]
                actual = tuple(
                    copies[field] for field in ("h2d_bytes", "d2h_bytes", "h2d_calls", "d2h_calls")
                )
                require(
                    all(type(value) is int for value in actual) and actual == expected,
                    f"wrong {method} transfer contract: {actual}",
                )
        for method in METHODS:
            checksum = record["checksum"][method]
            require(
                all(
                    type(checksum[key]) in (int, float) and math.isfinite(checksum[key])
                    for key in ("pnl", "costs", "turnover")
                ),
                "nonfinite output checksum",
            )
            require(bool(checksum["ordered_hash_fnv1a64"]), "missing ordered output hash")
        hashes = {record["checksum"][method]["ordered_hash_fnv1a64"] for method in GPU_METHODS}
        require(len(hashes) == 1, "GPU methods returned different operation outputs")
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError(f"Malformed pipeline record: {error}") from error
    return record


def summarize_records(records):
    """Use process-session medians and paired ratios, never pooled sample independence."""
    groups = defaultdict(list)
    for entry in records:
        record = entry["record"]
        workload = record["workload"]
        groups[(workload["paths"], workload["intervals"], workload["operations"])].append(entry)
    summary = []
    for (paths, intervals, operations), entries in sorted(groups.items()):
        require(
            len({entry["session"] for entry in entries}) == len(entries),
            "duplicate process session",
        )
        require(
            len({entry["record"]["workload"]["input_hash_fnv1a64"] for entry in entries}) == 1,
            "different inputs across sessions",
        )
        require(
            len(
                {
                    json.dumps(
                        (entry["record"]["device"], entry["record"]["build"]), sort_keys=True
                    )
                    for entry in entries
                }
            )
            == 1,
            "different device/build across sessions",
        )
        medians = [
            {
                method: statistics.median(
                    sample[f"{method}_ms"] for sample in entry["record"]["pipeline_samples"]
                )
                for method in METHODS
            }
            for entry in entries
        ]
        ratios = [row["copy_per_call"] / row["resident_session"] for row in medians]
        row = {
            "paths": paths,
            "intervals": intervals,
            "operations": operations,
            "process_sessions": len(entries),
            "per_process": [
                {"session": entry["session"], "medians_ms": values, "paired_session_speedup": ratio}
                for entry, values, ratio in zip(entries, medians, ratios, strict=True)
            ],
        }
        row.update(
            {
                f"{method}_median_ms": statistics.median(values[method] for values in medians)
                for method in METHODS
            }
        )
        row["paired_session_speedup_median"] = statistics.median(ratios)
        row["paired_session_speedup_range"] = [min(ratios), max(ratios)]
        summary.append(row)
    return summary


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    files = []
    for directory in (ROOT / "cuda", ROOT / "native"):
        files += [
            file
            for file in directory.rglob("*")
            if file.is_file()
            and not any(
                part in {"build", "__pycache__", ".build-local"}
                for part in file.relative_to(ROOT).parts
            )
            and (
                file.name == "CMakeLists.txt"
                or file.suffix in {".cu", ".cpp", ".hpp", ".py", ".in", ".toml", ".sh"}
            )
        ]
    return {str(file.relative_to(ROOT)): sha256(file) for file in sorted(files)}


def write_json(path, record):
    with Path(path).open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(args):
    output = Path(args.output).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError(f"output already exists: {output}") from error
    binary = Path(args.binary).resolve()
    require(binary.is_file(), f"missing binary: {binary}")
    for values, name in ((args.paths, "paths"), (args.operations, "operations")):
        require(
            len(set(values)) == len(values) and all(value > 0 for value in values),
            f"invalid or duplicate {name}",
        )
    require(all(value <= 256 for value in args.operations), "operations must be in 1..256")
    require(
        1 <= args.intervals <= 256 and 1 <= args.block_size <= 1024, "invalid intervals/block size"
    )
    require(
        args.sessions > 0 and args.repeats > 0 and args.warmups >= 0, "invalid sampling settings"
    )
    jobs = [
        (paths, operations, session)
        for paths in args.paths
        for operations in args.operations
        for session in range(args.sessions)
    ]
    random.Random(args.seed).shuffle(jobs)
    manifest = {
        "schema_version": 1,
        "kind": "resident accounting process-session comparison",
        "source_revision": args.source_revision,
        "source_sha256": source_hashes(),
        "binary_sha256": sha256(binary),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "settings": vars(args),
        "job_order": jobs,
        "primary_boundary": (
            "initial H2D upload, all K arithmetic kernels, all K outputs D2H, "
            "synchronization; capacity allocation excluded"
        ),
        "repeated_boundary": (
            "already uploaded inputs, all K kernels and all K outputs D2H, "
            "synchronization; initial upload excluded"
        ),
        "outputs": "all three per-path arrays for every operation; unchanged arithmetic",
        "limitations": [
            "fixed-input cost sensitivity workload, not GPU training or complete HPO",
            "one host; three process sessions are not a broad hardware study",
            "CPU reference does not cache common accounting terms",
            "residency also batches output copies and reduces synchronization",
        ],
    }
    write_json(output / "manifest.json", manifest)
    raw = output / "raw"
    raw.mkdir()
    records = []
    for paths, operations, session in jobs:
        name = f"paths-{paths}-ops-{operations}-session-{session}"
        destination = raw / f"{name}.json"
        command = [
            str(binary),
            "--mode",
            "pipeline",
            "--paths",
            str(paths),
            "--intervals",
            str(args.intervals),
            "--operations",
            str(operations),
            "--block-size",
            str(args.block_size),
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
            "--device",
            str(args.device),
            "--output",
            str(destination),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
        write_json(
            raw / f"{name}-execution.json",
            {
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
        require(completed.returncode == 0, f"pipeline process failed: {name}; see execution record")
        record = json.loads(destination.read_text())
        validate_record(
            record,
            paths=paths,
            intervals=args.intervals,
            operations=operations,
            repeats=args.repeats,
        )
        records.append({"session": session, "record": record})
        print(f"Validated {name}", flush=True)
    require(sha256(binary) == manifest["binary_sha256"], "binary changed during collection")
    require(source_hashes() == manifest["source_sha256"], "sources changed during collection")
    summary = {
        "schema_version": 1,
        "primary_comparison": "resident_session versus copy_per_call",
        "aggregation": "median of process medians; median/range of paired per-process ratios",
        "measurements": summarize_records(records),
        "records": len(records),
        "limitations": manifest["limitations"],
    }
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paths", type=int, nargs="+", default=[1, 256, 2048, 65536])
    parser.add_argument("--operations", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--intervals", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--source-revision")
    parser.add_argument("--timeout", type=float, default=180)
    try:
        run(parser.parse_args())
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
