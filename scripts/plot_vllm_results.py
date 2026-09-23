from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="assets/vllm_concurrency.png")
    args = parser.parse_args()
    result_dir = Path(args.results)
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(result_dir.glob("vllm_concurrency_*.json"))
    ]
    reports.sort(key=lambda item: item["concurrency"])
    concurrency = [item["concurrency"] for item in reports]
    throughput = [item["aggregate_output_tokens_per_second"] for item in reports]
    ttft = [item["ttft_ms"]["median"] for item in reports]
    ttft_p95 = [item["ttft_ms"]["p95"] for item in reports]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(concurrency, throughput, marker="o", linewidth=2.5, color="#176B87")
    axes[0].set(
        title="Continuous-batching throughput",
        xlabel="Concurrent requests",
        ylabel="Aggregate output tokens/s",
        xticks=concurrency,
    )
    axes[1].plot(concurrency, ttft, marker="o", linewidth=2.5, label="median", color="#176B87")
    axes[1].plot(concurrency, ttft_p95, marker="o", linewidth=2.5, label="p95", color="#E07A5F")
    axes[1].set(
        title="Streaming time to first token",
        xlabel="Concurrent requests",
        ylabel="TTFT (ms)",
        xticks=concurrency,
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("vLLM 0.30.0 — Qwen2.5-0.5B on RTX 4050")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
