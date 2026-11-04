"""Regenerate the diagnostic figure from the archived process-session summary."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = json.loads((HERE / "summary.json").read_text())["measurements"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
paths = [1, 256, 2048, 65536]
operations = [1, 8, 32]
selected = [row for row in rows if row["operations"] == 32]
for method, label, color in (
    ("copy_per_call", "Copy inputs for every operation", "#bf582b"),
    ("resident_session", "Resident session: upload once", "#176d9c"),
):
    values = [row[f"{method}_median_ms"] for row in selected]
    lows = [
        min(process["medians_ms"][method] for process in row["per_process"]) for row in selected
    ]
    highs = [
        max(process["medians_ms"][method] for process in row["per_process"]) for row in selected
    ]
    axes[0].errorbar(
        paths,
        values,
        yerr=[
            [v - lo for v, lo in zip(values, lows, strict=True)],
            [hi - v for v, hi in zip(values, highs, strict=True)],
        ],
        marker="o",
        capsize=3,
        label=label,
        color=color,
    )
axes[0].set(
    xscale="log",
    yscale="log",
    xlabel="Paths per operation",
    ylabel="Completed session (ms)",
    title="32 operations; every result returned",
)
axes[0].set_xticks(paths, ["1", "256", "2,048", "65,536"])
axes[0].legend(fontsize=8, loc="upper left")
for count, color in zip(paths, ("#667085", "#986038", "#168577", "#375ba8"), strict=True):
    group = [row for row in rows if row["paths"] == count]
    values = [row["paired_session_speedup_median"] for row in group]
    low = [
        value - row["paired_session_speedup_range"][0]
        for value, row in zip(values, group, strict=True)
    ]
    high = [
        row["paired_session_speedup_range"][1] - value
        for value, row in zip(values, group, strict=True)
    ]
    axes[1].errorbar(
        operations,
        values,
        yerr=[low, high],
        marker="o",
        capsize=3,
        color=color,
        label=f"{count:,} {'path' if count == 1 else 'paths'}",
    )
axes[1].axhline(1, color="#777777", linestyle="--", linewidth=1)
axes[1].set(
    xlabel="Operations reusing the inputs",
    ylabel="Copy-per-call / resident session time",
    title="Paired process-session speedup",
)
axes[1].set_xticks(operations)
axes[1].legend(fontsize=8)
for axis in axes:
    axis.grid(alpha=0.2)
figure.suptitle(
    "A100 float64 accounting: residency, batched readback, fewer synchronizations", fontsize=12
)
figure.text(
    0.5,
    0.025,
    "12 intervals; initial upload and all outputs included. "
    "Whiskers: min–max across 3 process sessions.\n"
    "Reusable capacity; context/allocation excluded. Development observations on one host.",
    ha="center",
    fontsize=8,
    color="#475467",
)
figure.tight_layout(rect=(0, 0.105, 1, 0.93))
figure.savefig(HERE / "session-performance.png", dpi=160)
plt.close(figure)
