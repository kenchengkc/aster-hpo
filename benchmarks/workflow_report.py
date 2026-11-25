"""Recompute full-budget NumPy and CUDA timings from a saved workflow comparison."""

import argparse
from pathlib import Path

from .common import validate_seeds
from .cuda_workflow import ROOT, _digest, _finite, _read, build_jobs, summarize, validate_summary

EVIDENCE = ROOT / "docs/evidence/cuda-launch-2026-11-04"


def _full_budget_records(manifest, records):
    protocol = manifest["protocol"]
    validate_seeds(protocol["seeds"])
    if not protocol["markets"] or not protocol.get("include_eager"):
        raise ValueError("The protocol must include workloads and both CUDA implementations")
    digest = _digest(protocol)
    if digest != manifest["protocol_sha256"]:
        raise ValueError("The manifest protocol differs from its saved fingerprint")
    expected = {job["id"]: job for job in build_jobs(protocol) if job["scheduler"] == "full"}
    selected = []
    for record in records:
        job = record["job"]
        if job["scheduler"] != "full":
            continue
        name = job["id"]
        if expected.pop(name, None) != job:
            raise ValueError(f"Duplicate or unexpected full-budget campaign {name}")
        if (
            record["protocol_sha256"] != digest
            or record["source_sha256"] != manifest["source"]["sha256"]
        ):
            raise ValueError(f"{name} differs from the manifest protocol or source")
        for field in ("workflow_seconds", "search_seconds", "audit_seconds"):
            _finite(record["timings"].get(field), f"{name} {field}", positive=True)
        _finite(record.get("process_seconds"), f"{name} process_seconds", positive=True)
        if record["process_seconds"] < record["timings"]["workflow_seconds"]:
            raise ValueError(f"{name} process_seconds is shorter than workflow_seconds")
        selected.append(record)
    if expected:
        raise ValueError(f"Missing full-budget campaigns: {', '.join(sorted(expected))}")
    return selected


def render_report(manifest, records):
    protocol = manifest["protocol"]
    groups = validate_summary(summarize(_full_budget_records(manifest, records), protocol))
    by_workload = {
        (category, group["workload"]): group
        for category in ("full_budget", "cpu_four_worker_control", "cuda_optimization")
        for group in groups[category]
    }
    lines = [
        f"Full-budget search and independent winner audit ({len(protocol['seeds'])} seeds)",
        "Times are median seconds. Speedup is each row's time divided by optimized CUDA time",
        "for the same workload and seed, then summarized by the median.",
        "",
        f"{'Train / eval paths':<20}{'Implementation':<20}"
        f"{'Workflow (s)':>13}{'Speedup':>10}{'Process (s)':>13}{'Speedup':>10}",
    ]
    for workload, market in protocol["markets"].items():
        paths = f"{market['batch_paths']:,} / {market['evaluation_paths']:,}"
        for category, prefix, label in (
            ("full_budget", "numpy", "NumPy (1 worker)"),
            ("cpu_four_worker_control", "numpy", "NumPy (4 workers)"),
            ("cuda_optimization", "eager", "Eager CUDA"),
            ("cuda_optimization", "optimized", "Optimized CUDA"),
        ):
            group = by_workload[category, workload]
            row = f"{paths:<20}{label:<20}"
            for scope in ("workflow", "process"):
                seconds = group[f"{prefix}_{scope}_seconds"]["median"]
                speedup = 1.0
                if prefix != "optimized":
                    field = f"{scope}_speedup"
                    for pair in group["pairs"]:
                        _finite(
                            pair[field], f"{workload} {pair['seed']} {label} {field}", positive=True
                        )
                    speedup = group[field]["median"]
                row += f"{seconds:13.3f}{speedup:9.3f}×"
            lines.append(row)
    lines.extend(
        [
            "",
            "Workflow includes numerical imports, backend setup, search, winner selection,",
            "audits, device completion and result writes. The parent process clock also",
            "includes interpreter startup, timing-file publication and interpreter exit.",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=EVIDENCE,
        help="directory with manifest.json and records.json",
    )
    args = parser.parse_args()
    try:
        manifest = _read(args.directory / "manifest.json")
        records = _read(args.directory / "records.json")
        print(render_report(manifest, records))
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
        parser.exit(1, f"Cannot report on {args.directory}: {error}\n")


if __name__ == "__main__":
    main()
