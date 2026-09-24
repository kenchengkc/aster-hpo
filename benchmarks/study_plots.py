"""Standalone research figures regenerated from validated raw artifacts."""

import json
import math
import statistics
from pathlib import Path

from .financial_report import regenerate

LABELS = {
    "serial_full": "Serial full budget",
    "parallel_full": "Parallel full budget",
    "asha": "ASHA",
    "race": "Replication race",
}
COLORS = {
    "serial_full": "#777777",
    "parallel_full": "#35618c",
    "asha": "#c48527",
    "race": "#248574",
}


def mean_interval(values):
    """Return a nominal 95% t half-width, or no interval for insufficient runs."""
    from scipy.stats import t

    if len(values) < 2:
        return None
    se = statistics.stdev(values) / math.sqrt(len(values))
    return float(t.ppf(0.975, len(values) - 1)) * se


def plot_study(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output = Path(output)
    report = regenerate(output)
    records = json.loads((output / "records.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    for field, xlabel, filename in [
        ("median_search_wall_seconds", "Median search wall time (seconds)", "quality_time.png"),
        (
            "mean_objective_cpu_seconds",
            "Mean returned objective CPU time (seconds)",
            "compute_quality.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for method, row in report["methods"].items():
            means = [
                statistics.mean(r["audit_losses"])
                for r in records
                if r["method"] == method and r["audit_losses"] is not None
            ]
            if not means:
                continue
            ax.errorbar(
                row[field],
                row["mean_audited_loss"],
                yerr=mean_interval(means),
                fmt="o",
                capsize=4,
                color=COLORS[method],
                label=LABELS[method] + (" (one run; no interval)" if len(means) == 1 else ""),
            )
        ax.set(
            xlabel=xlabel,
            ylabel="Mean independently audited hedging MSE",
            title="Fixed-budget endpoints — lower and left are better",
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.text(
            0.02,
            -0.04,
            "95% t intervals when ≥2 optimizer seeds; not time-to-target curves.",
            fontsize=9,
        )
        fig.savefig(output / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    scaling = report["scaling"]
    if scaling:
        workers = sorted({r["workers"] for r in scaling})
        baseline = {r["seed"]: r["wall_seconds"] for r in scaling if r["workers"] == 1}
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        medians = []
        for worker in workers:
            ratios = [
                baseline[r["seed"]] / r["wall_seconds"] for r in scaling if r["workers"] == worker
            ]
            medians.append(statistics.median(ratios))
            ax.scatter([worker] * len(ratios), ratios, alpha=0.4, color="#248574")
        ax.plot(workers, medians, color="#248574", marker="o", label="Median measured speedup")
        ax.plot(workers, workers, linestyle="--", color="#999999", label="Ideal scaling")
        ax.set(
            xlabel="Workers",
            ylabel="Speedup relative to one worker",
            title="Fixed simulation work; no adaptive pruning",
            xticks=workers,
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        fig.savefig(output / "scaling.png", dpi=160)
        plt.close(fig)

    example = next(spec for spec in manifest["campaigns"] if spec["method"] == "race")
    path = output / "campaigns" / example["id"] / "search.json"
    if not path.exists():
        return
    result = json.loads(path.read_text())
    observations = [
        (trial["status"], observation)
        for trial in result["trials"]
        for observation in trial["observations"]
    ]
    pids = sorted({o["worker_pid"] for _, o in observations})
    if not observations:
        return
    start = min(event["at"] for event in result["metadata"]["events"] if event["kind"] == "submit")
    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    for status, observation in observations:
        color = "#248574" if status == "complete" else "#b5bcc4"
        ax.broken_barh(
            [(observation["started_at"] - start, observation["wall_seconds"])],
            (pids.index(observation["worker_pid"]) - 0.35, 0.7),
            facecolors=color,
        )
    ax.set(
        yticks=list(range(len(pids))),
        yticklabels=[f"Worker {i + 1}" for i in range(len(pids))],
        xlabel="Seconds since first submission",
        title="Replication race worker timeline",
    )
    ax.legend(
        handles=[
            Patch(color="#248574", label="Completed candidate"),
            Patch(color="#b5bcc4", label="Pruned candidate"),
        ],
        fontsize=9,
    )
    fig.savefig(output / "worker_timeline.png", dpi=160)
    plt.close(fig)
