from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkConfig:
    model_id: str
    device: str
    dtype: str
    context_lengths: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    new_tokens: int
    warmup_runs: int
    measured_runs: int
    seed: int
    backends: tuple[str, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "BenchmarkConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            model_id=payload["model_id"],
            device=payload["device"],
            dtype=payload["dtype"],
            context_lengths=tuple(payload["context_lengths"]),
            batch_sizes=tuple(payload["batch_sizes"]),
            new_tokens=int(payload["new_tokens"]),
            warmup_runs=int(payload["warmup_runs"]),
            measured_runs=int(payload["measured_runs"]),
            seed=int(payload["seed"]),
            backends=tuple(payload["backends"]),
        )

    def validate(self) -> None:
        if self.new_tokens < 2:
            raise ValueError("new_tokens must be at least 2 to measure decode latency")
        if self.measured_runs < 1:
            raise ValueError("measured_runs must be positive")
        if any(value < 1 for value in (*self.context_lengths, *self.batch_sizes)):
            raise ValueError("context lengths and batch sizes must be positive")
        allowed = {"transformers_kv_cache", "transformers_no_cache"}
        unknown = set(self.backends) - allowed
        if unknown:
            raise ValueError(f"unsupported backends: {sorted(unknown)}")
