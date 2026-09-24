#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / "results" / "gateway_simulation.json").read_text(encoding="utf-8"))
runs = report["runs"]
controlled = [run for run in runs if run["admission_control"]]
uncontrolled = [run for run in runs if not run["admission_control"]]
labels = [run["policy"].replace("_", "\n") for run in controlled]
x = list(range(len(labels)))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
width = 0.36
for axis, metric, title, ylabel in [
    (axes[0], "accepted_slo_rate_pct", "Accepted requests meeting SLO", "%"),
    (axes[1], "p95", "End-to-end p95 latency", "ms"),
    (axes[2], "mean_queue_depth", "Mean queue depth", "requests"),
]:
    if metric == "p95":
        values_without = [run["latency_ms"][metric] for run in uncontrolled]
        values_with = [run["latency_ms"][metric] for run in controlled]
    else:
        values_without = [run[metric] for run in uncontrolled]
        values_with = [run[metric] for run in controlled]
    axis.bar([i - width / 2 for i in x], values_without, width, label="No admission")
    axis.bar([i + width / 2 for i in x], values_with, width, label="SLO admission")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(x, labels)
    axis.grid(axis="y", alpha=0.25)

axes[0].legend(loc="lower right")
fig.suptitle("Deterministic gateway overload simulation — not a GPU benchmark", fontsize=14)
fig.tight_layout()
output = ROOT / "assets" / "gateway_simulation.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print(output)
