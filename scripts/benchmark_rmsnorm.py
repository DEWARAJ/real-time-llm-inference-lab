from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from inference_lab.kernels import rms_norm_reference
from inference_lab.triton_rmsnorm import triton_rms_norm


def time_cuda(function, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def stats(samples: list[float]) -> dict[str, float]:
    return {
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "minimum_ms": float(np.min(samples)),
        "maximum_ms": float(np.max(samples)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", default="results/triton_rmsnorm.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(11)
    shapes = [(1024, 896), (4096, 896), (1024, 3584)]
    experiments = []
    for shape in shapes:
        x = torch.randn(*shape, device="cuda", dtype=torch.float16)
        weight = torch.randn(shape[-1], device="cuda", dtype=torch.float16)
        expected = rms_norm_reference(x, weight)
        actual = triton_rms_norm(x, weight)
        max_error = float((expected - actual).abs().max().item())
        if max_error > 1e-2:
            raise AssertionError(f"Triton parity error {max_error} exceeds 1e-2")
        reference = time_cuda(
            lambda: rms_norm_reference(x, weight), args.warmup, args.repeats
        )
        triton_samples = time_cuda(
            lambda: triton_rms_norm(x, weight), args.warmup, args.repeats
        )
        reference_stats = stats(reference)
        triton_stats = stats(triton_samples)
        experiments.append(
            {
                "shape": list(shape),
                "dtype": "float16",
                "max_abs_error": max_error,
                "pytorch": reference_stats,
                "triton": triton_stats,
                "median_speedup": (
                    reference_stats["median_ms"] / triton_stats["median_ms"]
                ),
            }
        )
    report = {
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "experiments": experiments,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
