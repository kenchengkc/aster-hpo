"""Regenerate full-budget workflow timings and independent audit diagnostics."""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
records = json.loads((HERE / "records.json").read_text())
summary = json.loads((HERE / "summary.json").read_text())
workloads = ("tiny", "scaled")
figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
plt.rcParams.update({"font.size": 10})
for offset, backend, workers, label, color in (
    (-0.25, "numpy", 1, "NumPy: 1 worker", "#bf582b"),
    (0.0, "numpy", 4, "NumPy: 4 workers", "#9b884e"),
    (0.25, "cuda", 1, "CUDA: 1 worker", "#176d9c"),
):
    values = []
    errors = [[], []]
    for workload in workloads:
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
        errors[0].append(value - min(times))
        errors[1].append(max(times) - value)
    axes[0].bar(
        np.arange(2) + offset,
        values,
        width=0.23,
        color=color,
        label=label,
        yerr=errors,
        capsize=3,
        alpha=0.92,
    )
axes[0].set_xticks(
    [0, 1], ["128 training / 2,048 evaluation", "4,096 training / 32,768 evaluation"], fontsize=8
)
axes[0].set(
    ylabel="Full-budget search + audit workflow (s)",
    title="Four candidates × four replications + three audits",
)
axes[0].legend(fontsize=8)
for offset, scheduler, color in ((-0.12, "full_budget", "#176d9c"), (0.12, "adaptive", "#168577")):
    for index, workload in enumerate(workloads):
        group = next(row for row in summary[scheduler] if row["workload"] == workload)
        differences = [pair["audit_mean_difference"] for pair in group["pairs"]]
        axes[1].scatter(
            [index + offset] * len(differences),
            differences,
            color=color,
            marker="o" if scheduler == "full_budget" else "D",
            s=40,
            label="Full budget"
            if index == 0 and scheduler == "full_budget"
            else "Replication race"
            if index == 0
            else None,
        )
axes[1].axhline(0, color="#777777", linestyle="--", linewidth=1)
axes[1].set_xticks([0, 1], ["Original workload", "Larger workload"])
axes[1].set(
    ylabel="Audited MSE difference: CUDA − NumPy",
    title="Independent locked-winner audits; one CPU worker",
)
axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axes[1].legend(fontsize=8)
for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
figure.suptitle("A100 float64 neural hedging: actual simulation and HPO", fontsize=12)
figure.text(
    0.5,
    0.025,
    "Three optimizer seeds per setting; bars: medians, whiskers: min–max. "
    "Fresh processes, shared compiler cache.\n"
    "Workflow includes backend setup, search, independent audit and result writes. "
    "One host; descriptive development evidence.",
    ha="center",
    fontsize=8,
    color="#475467",
)
figure.tight_layout(rect=(0, 0.11, 1, 0.94))
figure.savefig(HERE / "workflow-performance.png", dpi=160)
plt.close(figure)
