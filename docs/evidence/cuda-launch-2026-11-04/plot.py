"""Regenerate launch diagnostics and unprofiled full-workflow comparisons."""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
profiles = json.loads((HERE / "profiles-summary.json").read_text())["profiles"]
records = json.loads((HERE / "records.json").read_text())
figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for offset, name, label, color in (
    (-0.18, "eager", "Previous CUDA", "#9b884e"),
    (0.18, "optimized", "Fused CUDA", "#176d9c"),
):
    values = [
        next(
            row["kernel_executions"]
            for row in profiles
            if row["shape"] == shape and row["implementation"] == name
        )
        for shape in ("small", "scaled")
    ]
    bars = axes[0].bar(np.arange(2) + offset, values, width=0.34, color=color, label=label)
    for bar, value in zip(bars, values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.1,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
axes[0].set(
    yscale="log",
    ylabel="Kernel executions in one objective",
    title="Warm diagnostic: 64 epochs, width 8",
)
axes[0].set_xticks([0, 1], ["128 training paths", "4,096 training paths"])
axes[0].legend(fontsize=8, loc="lower left")
axes[0].set_ylim(top=max(row["kernel_executions"] for row in profiles) * 2.5)
for offset, backend, workers, label, color in (
    (-0.30, "numpy", 1, "NumPy: 1 worker", "#bf582b"),
    (-0.10, "numpy", 4, "NumPy: 4 workers", "#ad8e5b"),
    (0.10, "cuda_eager", 1, "Previous CUDA", "#9b884e"),
    (0.30, "cuda", 1, "Fused CUDA", "#176d9c"),
):
    values, low, high = [], [], []
    for workload in ("tiny", "scaled"):
        times = [
            row["timings"]["workflow_seconds"]
            for row in records
            if row["job"]["workload"] == workload
            and row["job"]["scheduler"] == "full"
            and row["job"]["backend"] == backend
            and row["job"]["workers"] == workers
        ]
        value = statistics.median(times)
        values.append(value)
        low.append(value - min(times))
        high.append(max(times) - value)
    axes[1].bar(
        np.arange(2) + offset,
        values,
        width=0.18,
        color=color,
        label=label,
        yerr=[low, high],
        capsize=2,
    )
axes[1].set(
    yscale="log",
    ylabel="Complete search + audit workflow (s)",
    title="Full budget: four candidates × four replications",
)
axes[1].set_xticks([0, 1], ["Original workload", "Larger workload"])
axes[1].legend(fontsize=8)
for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
figure.suptitle("A100 float64 hedging: fewer launches, complete-workflow evidence", fontsize=12)
figure.text(
    0.5,
    0.025,
    "Left: Nsight diagnostic, not benchmark timings. Right: medians and min–max "
    "across 3 paired optimizer seeds.\n"
    "Fresh processes, shared compilation cache; setup, transfers, search, independent "
    "audits and result writes included.",
    ha="center",
    fontsize=8,
    color="#475467",
)
figure.tight_layout(rect=(0, 0.11, 1, 0.94))
figure.savefig(HERE / "launch-performance.png", dpi=160)
plt.close(figure)
