"""Validate raw experiment artifacts and regenerate the financial report."""

import json
import statistics
from pathlib import Path

from aster_hpo import Replication

from .analysis import analyze_campaigns
from .common import write_json
from .protocol import digest


def _load(path):
    return json.loads(path.read_text())


def _record(folder, spec, protocol):
    if _load(folder / "run.json") != spec:
        raise ValueError("Campaign identity mismatch")
    if (folder / "failure.json").exists():
        failure = _load(folder / "failure.json")
        search, audit = failure["search"], failure["audit"]
        return {
            **spec,
            "audit_losses": None,
            "error": failure["error"],
            "search_wall_seconds": search["wall_seconds"],
            "objective_cpu_seconds": sum(o["cpu_seconds"] for o in search["observations"]),
            "replications": len(search["observations"]),
            "audit_wall_seconds": audit["wall_seconds"],
            "audit_cpu_seconds": sum(o["cpu_seconds"] for o in audit["observations"]),
            "allocated_worker_seconds": spec["workers"] * search["wall_seconds"],
            "unknown_attempts": sum(
                a["status"] != "accepted" for phase in (search, audit) for a in phase["attempts"]
            ),
        }
    search = _load(folder / "search.json")
    checked = _load(folder / "audit.json")
    winner = _load(folder / "winner.json")
    selected = next(t for t in search["trials"] if t["id"] == search["best_trial_id"])
    if winner != selected["config"] or selected["status"] != "complete":
        raise ValueError("Locked winner does not match completed search")
    observations = checked["observations"]
    if len(observations) != protocol["audit_replications"]:
        raise ValueError("Incomplete audit")
    for index, observation in enumerate(observations):
        expected = vars(Replication.create(spec["seed"], winner, index, "audit", True))
        if observation["replication"] != expected or observation["error"] is not None:
            raise ValueError("Audit identity or outcome mismatch")
    return {
        **spec,
        "audit_losses": [o["loss"] for o in observations],
        "search_wall_seconds": search["wall_seconds"],
        "objective_cpu_seconds": search["objective_cpu_seconds"],
        "replications": sum(len(t["observations"]) for t in search["trials"]),
        "audit_wall_seconds": checked["wall_seconds"],
        "audit_cpu_seconds": sum(o["cpu_seconds"] for o in observations),
        "allocated_worker_seconds": search["metadata"]["allocated_worker_seconds"],
        "unknown_attempts": (
            search["metadata"]["unmeasured_attempts"] + checked["metadata"]["unmeasured_attempts"]
        ),
    }


def regenerate(output):
    output = Path(output)
    manifest = _load(output / "manifest.json")
    protocol = manifest["protocol"]
    if manifest["schema_version"] != 1 or digest(protocol) != manifest["protocol_sha256"]:
        raise ValueError("Invalid protocol manifest")
    records = [
        _record(output / "campaigns" / spec["id"], spec, protocol) for spec in manifest["campaigns"]
    ]
    report = analyze_campaigns(records, margin=protocol["quality_margin"])
    scaling = []
    for spec in manifest["scaling"]:
        result = _load(output / "scaling" / spec["id"] / "result.json")
        if any(t["status"] != "complete" for t in result["trials"]):
            raise ValueError("Incomplete fixed-workload scaling run")
        scaling.append(
            {
                **spec,
                "wall_seconds": result["wall_seconds"],
                "objective_cpu_seconds": result["objective_cpu_seconds"],
            }
        )
    report["scaling"] = scaling
    report["protocol_sha256"] = manifest["protocol_sha256"]
    write_json(output / "records.json", records)
    write_json(output / "summary.json", report)
    lines = [
        f"# {'CUDA' if protocol.get('backend', 'numpy').startswith('cuda') else 'CPU'} "
        "hedging allocation study",
        "",
        "Fixed candidate and replication caps; endpoint quality versus elapsed search time.",
        "These are synthetic-market results, not live trading performance.",
        "",
        f"Independent optimizer seeds: {len(protocol['seeds'])}. "
        f"Audit replications per locked winner: {protocol['audit_replications']}.",
        f"Prespecified excess-MSE margin: {protocol['quality_margin']:.8g} "
        f"({100 * protocol['margin_fraction']:g}% of disjoint pilot reference).",
        "",
        "| Method | Failed | Mean audited MSE | Median search seconds | Mean objective CPU seconds "
        "| Mean replications | Median speedup vs parallel full |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def number(value):
        return "unavailable" if value is None else f"{value:.6g}"

    for method, row in report["methods"].items():
        lines.append(
            f"| {method} | {row['failed_runs']} | {number(row['mean_audited_loss'])} "
            f"| {number(row['median_search_wall_seconds'])} "
            f"| {number(row['mean_objective_cpu_seconds'])} "
            f"| {number(row['mean_replications'])} | {number(row['median_speedup'])} |"
        )
    lines.extend(["", "## Independent quality comparison", ""])
    for method, comparison in report["comparisons"].items():
        lines.append(
            f"- {method}: mean excess MSE {number(comparison['mean_difference'])}; "
            f"one-sided 95% upper bound {number(comparison['upper_bound'])}; "
            f"status: **{comparison['status']}**."
        )
    lines.extend(
        [
            "",
            "The primary comparison is racing versus parallel full budget. Other method",
            "comparisons are descriptive, with nominal intervals and no multiplicity correction.",
            "The t analysis assumes independent optimizer runs and an adequate approximation for",
            "their mean differences; it is not a sequential pruning guarantee.",
            "Audit paths and training replications are nested within optimizer runs, not counted",
            "as extra independent runs. Inconclusive intervals do not establish equivalence.",
            "A degenerate comparison has zero sample variance: its observed difference is",
            "reported, but no inferential bound or non-inferiority conclusion is supplied.",
            "",
            "## Cost and scope",
            "",
            "Search time includes process startup, journal writes, and idle capacity. Audit costs",
            "are reported separately below. Returned objective CPU time excludes coordinator and",
            "startup overhead. Missing attempt cost is explicitly counted in summary.json.",
            "Objective CPU seconds measure host CPU consumption, not GPU execution cost.",
            "The speedup bootstrap interval resamples whole paired optimizer seeds.",
            "These endpoints are not time-to-target curves or equal-wall-budget comparisons.",
            "",
        ]
    )
    for method, row in report["methods"].items():
        lines.append(
            f"- {method}: mean audit wall {row['mean_audit_wall_seconds']:.4g}s; "
            f"mean audit CPU {row['mean_audit_cpu_seconds']:.4g}s; "
            f"unmeasured attempts {row['unmeasured_attempts']}; "
            f"speedup 95% bootstrap interval {row['speedup_bootstrap_95_interval']}."
        )
    if scaling:
        lines.extend(
            [
                "",
                "## Fixed-workload scaling",
                "",
                f"{protocol['scaling_count']} candidates × "
                f"{protocol['scaling_replications']} replications, "
                f"{len(protocol['scaling_seeds'])} seeds; no pruning.",
            ]
        )
        for workers in protocol["scaling_workers"]:
            times = [row["wall_seconds"] for row in scaling if row["workers"] == workers]
            lines.append(f"- {workers} workers: median {statistics.median(times):.4g}s.")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report
