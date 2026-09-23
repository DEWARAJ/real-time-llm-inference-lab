from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/transformers_benchmark.json")
    parser.add_argument("--output", default="assets/benchmark_summary.png")
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    experiments = report["experiments"]
    backends = sorted({item["backend"] for item in experiments})
    batches = sorted({item["batch_size"] for item in experiments})
    fig, axes = plt.subplots(len(batches), 2, figsize=(11, 7.5), squeeze=False)
    for row_index, batch in enumerate(batches):
        batch_rows = [item for item in experiments if item["batch_size"] == batch]
        for backend in backends:
            rows = sorted(
                (item for item in batch_rows if item["backend"] == backend),
                key=lambda item: item["context_length"],
            )
            label = backend.replace("transformers_", "").replace("_", " ")
            axes[row_index, 0].plot(
                [row["context_length"] for row in rows],
                [row["summary"]["tpot_ms"]["median"] for row in rows],
                marker="o",
                linewidth=2,
                label=label,
            )
            axes[row_index, 1].plot(
                [row["context_length"] for row in rows],
                [row["summary"]["output_tokens_per_second"]["median"] for row in rows],
                marker="o",
                linewidth=2,
                label=label,
            )
        axes[row_index, 0].set(
            title=f"Batch {batch}: decode latency",
            xlabel="Context tokens",
            ylabel="Median TPOT (ms)",
        )
        axes[row_index, 1].set(
            title=f"Batch {batch}: output throughput",
            xlabel="Context tokens",
            ylabel="Output tokens/s",
        )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(f"KV-cache reuse on {report['config']['model_id']}", fontsize=16)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
