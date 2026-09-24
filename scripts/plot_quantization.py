from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/quantization.json")
    parser.add_argument("--output", default="assets/quantization_tradeoff.png")
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    modes = [item["mode"].upper() for item in report["experiments"]]
    footprint = [item["model_footprint_mib"] for item in report["experiments"]]
    throughput = [item["output_tokens_per_second"] for item in report["experiments"]]
    perplexity = [item["quality_perplexity"] for item in report["experiments"]]
    colors = ["#176B87", "#E07A5F"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    panels = [
        (footprint, "Model footprint", "MiB"),
        (throughput, "Generation throughput", "Output tokens/s"),
        (perplexity, "Sanity-corpus perplexity", "Perplexity"),
    ]
    for axis, (values, title, ylabel) in zip(axes, panels):
        bars = axis.bar(modes, values, color=colors, width=0.62)
        axis.set(title=title, ylabel=ylabel)
        axis.grid(axis="y", alpha=0.2)
        axis.bar_label(bars, fmt="%.2f", padding=3)
        axis.set_ylim(0, max(values) * 1.18)
    fig.suptitle("BitsAndBytes NF4 tradeoff — Qwen2.5-0.5B on RTX 4050")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
