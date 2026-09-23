from __future__ import annotations

import math
from statistics import mean, median
from typing import Iterable


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Iterable[float]) -> dict[str, float]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": mean(samples),
        "median": median(samples),
        "p95": percentile(samples, 0.95),
        "minimum": min(samples),
        "maximum": max(samples),
    }


def summarize_runs(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("at least one run is required")
    return {
        "ttft_ms": summarize(run["ttft_ms"] for run in runs),
        "tpot_ms": summarize(run["tpot_ms"] for run in runs),
        "end_to_end_ms": summarize(run["end_to_end_ms"] for run in runs),
        "output_tokens_per_second": summarize(
            run["output_tokens_per_second"] for run in runs
        ),
        "peak_memory_mib": summarize(run["peak_memory_mib"] for run in runs),
    }
