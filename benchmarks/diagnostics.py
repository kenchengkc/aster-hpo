"""Run: python -m benchmarks.diagnostics --output results/diagnostics."""

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from aster_hpo import FullBudget, ReplicationRace, Study

from .common import create_experiment, validate_seeds, write_json
from .synthetic import CASES, SyntheticObjective, expected_loss, pool

DEFAULT_SEEDS = tuple(range(20))


def summarize_payload(result, case):
    trials = result["trials"]
    means = {t["id"]: expected_loss(case, t["config"]) for t in trials}
    optimum = min(means.values())
    complete = [t for t in trials if t["status"] == "complete"]
    winner = min(
        complete,
        key=lambda t: (statistics.mean(o["loss"] for o in t["observations"]), t["id"]),
        default=None,
    )
    best = [t for t in trials if means[t["id"]] == optimum]
    return {
        "winner": winner["config"] if winner else None,
        "pool_regret": means[winner["id"]] - optimum if winner else None,
        "continuous_regret": means[winner["id"]] if winner else None,
        "optimum_count": len(best),
        "any_optimum_pruned": any(t["status"] == "pruned" for t in best),
        "all_optima_pruned": all(t["status"] == "pruned" for t in best),
        "failed_trials": sum(t["status"] == "failed" for t in trials),
        "replications": sum(len(t["observations"]) for t in trials),
    }


def summarize_result(result, case):
    return summarize_payload(asdict(result), case)


def aggregate(records):
    groups = defaultdict(list)
    pairs = defaultdict(dict)
    for record in records:
        group = (record["case"], record["shared"], record["order"])
        groups[(*group, record["method"])].append(record)
        pairs[(*group, record["seed"])][record["method"]] = record
    rows = []
    for (case, shared, order, method), values in sorted(groups.items()):
        regret = [v["pool_regret"] for v in values if v["pool_regret"] is not None]
        rows.append(
            {
                "case": case,
                "shared": shared,
                "order": order,
                "method": method,
                "runs": len(values),
                "failed_runs": len(values) - len(regret),
                "mean_replications": statistics.mean(v["replications"] for v in values),
                "mean_pool_regret": statistics.mean(regret) if regret else None,
                "optimum_pruned_runs": sum(v["any_optimum_pruned"] for v in values),
            }
        )
    paired = defaultdict(list)
    for (case, shared, order, _seed), methods in sorted(pairs.items()):
        baseline = methods["full_budget"]["pool_regret"]
        race = methods["race"]["pool_regret"]
        paired[(case, shared, order)].append(
            None if baseline is None or race is None else race - baseline
        )
    return {
        "interpretation": "synthetic diagnostic",
        "runs": len(records),
        "groups": rows,
        "paired": [
            {
                "case": case,
                "shared": shared,
                "order": order,
                "independent_seeds": len(values),
                "failed_pairs": values.count(None),
                "differences": values,
            }
            for (case, shared, order), values in sorted(paired.items())
        ],
    }


def regenerate(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["kind"] != "synthetic" or manifest["schema_version"] != 1:
        raise ValueError("Unsupported diagnostic manifest")
    records = []
    for spec in manifest["runs"]:
        artifact = json.loads((output / "runs" / (spec["id"] + ".json")).read_text())
        if artifact["run"] != spec:
            raise ValueError(f"Run identity mismatch: {spec['id']}")
        if artifact["result"] is None:
            summary = {
                "pool_regret": None,
                "replications": artifact["accepted_replications"],
                "any_optimum_pruned": False,
            }
        else:
            summary = summarize_payload(artifact["result"], spec["case"])
        records.append({**spec, **summary})
    report = aggregate(records)
    write_json(output / "summary.json", report)
    lines = [
        "# Synthetic scheduler diagnostics",
        "",
        "Cheap known-mean objectives: these results do not establish practical speedup.",
        "Regret is measured against the best configuration in each frozen pool.",
        "Order permutations are sensitivity checks, not independent optimizer runs.",
        "",
        "| Case | Shared | Order | Method | Runs | Failed | Mean replications | Mean regret |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["groups"]:
        quality = (
            "unavailable" if row["mean_pool_regret"] is None else f"{row['mean_pool_regret']:.6g}"
        )
        lines.append(
            f"| {row['case']} | {row['shared']} | {row['order']} | {row['method']} "
            f"| {row['runs']} | {row['failed_runs']} | {row['mean_replications']:.1f} | {quality} |"
        )
    lines.extend(
        [
            "",
            "Positive paired differences in summary.json mean racing selected a worse",
            "configuration than full-budget search. Failures remain null, not zero.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report


def run_suite(output, *, seeds=DEFAULT_SEEDS, workers=1, count=12, checkpoints=(3, 9, 27)):
    seeds = validate_seeds(seeds)
    race = ReplicationRace(checkpoints)
    Study(pool(seeds[0], count), workers=workers, scheduler=race)
    specs = [
        {
            "id": f"{case}-{seed}-{shared}-{order}-{method}",
            "case": case,
            "seed": seed,
            "shared": shared,
            "order": order,
            "method": method,
        }
        for case in CASES
        for seed in seeds
        for shared in (True, False)
        for order in ("forward", "reverse")
        for method in ("full_budget", "race")
    ]
    output = create_experiment(
        output,
        {
            "schema_version": 1,
            "kind": "synthetic",
            "workers": workers,
            "count": count,
            "seeds": seeds,
            "scheduler": asdict(race),
            "runs": specs,
        },
    )
    for spec in specs:
        configs = pool(spec["seed"], count)
        if spec["order"] == "reverse":
            configs.reverse()
        scheduler = FullBudget(race.maximum) if spec["method"] == "full_budget" else race
        # Abort on infrastructure failure, leaving an explicitly incomplete manifest.
        result = Study(
            configs,
            workers=workers,
            seed=spec["seed"],
            scheduler=scheduler,
            shared_scenarios=spec["shared"],
        ).optimize(SyntheticObjective(spec["case"]))
        write_json(
            output / "runs" / (spec["id"] + ".json"), {"run": spec, "result": asdict(result)}
        )
    return regenerate(output)


def plot_report(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    report = regenerate(output)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for row in report["groups"]:
        if row["order"] != "forward" or row["shared"] is not True:
            continue
        axes[0].scatter(
            row["mean_replications"],
            row["mean_pool_regret"],
            marker="o" if row["method"] == "race" else "x",
            label=f"{row['case']} / {row['method']}",
        )
    axes[0].set(xlabel="Mean replications", ylabel="Mean pool regret", title="Shared scenarios")
    axes[0].legend(fontsize=6)
    rows = [r for r in report["paired"] if r["order"] == "forward" and r["shared"]]
    axes[1].boxplot(
        [[v for v in r["differences"] if v is not None] for r in rows],
        tick_labels=[r["case"] for r in rows],
    )
    axes[1].axhline(0, color="grey", linewidth=1)
    axes[1].set(ylabel="Race regret − full-budget regret", title="Paired optimizer seeds")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Synthetic diagnostics — not a practical speed benchmark")
    fig.savefig(output / "diagnostics.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--configs", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    report = (
        regenerate(args.output)
        if args.report_only
        else run_suite(
            args.output, seeds=tuple(range(args.seeds)), count=args.configs, workers=args.workers
        )
    )
    if args.plots:
        plot_report(args.output)
    print(f"{report['runs']} diagnostic runs; report: {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
