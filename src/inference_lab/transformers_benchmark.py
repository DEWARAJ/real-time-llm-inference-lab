from __future__ import annotations

import json
import platform
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BenchmarkConfig
from .metrics import summarize_runs


PROMPT_TEXT = (
    "Efficient inference requires measuring the complete system before optimizing it. "
    "Memory movement, cache layout, batching, and model execution all affect latency. "
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def _make_batch(tokenizer, context_length: int, batch_size: int, device):
    text = PROMPT_TEXT * max(2, context_length // 16 + 1)
    encoded = tokenizer(
        [text] * batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=context_length,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if input_ids.shape[1] < context_length:
        pad_count = context_length - input_ids.shape[1]
        filler = input_ids[:, -1:].expand(batch_size, pad_count)
        input_ids = torch.cat((input_ids, filler), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(filler, dtype=attention_mask.dtype)), dim=1
        )
    return input_ids.to(device), attention_mask.to(device)


@torch.inference_mode()
def _decode_once(model, input_ids, attention_mask, new_tokens: int, use_cache: bool):
    device = input_ids.device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    _sync(device)
    start = time.perf_counter()
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)
    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    _sync(device)
    ttft_ms = (time.perf_counter() - start) * 1000.0

    generated = torch.cat((input_ids, next_token), dim=1)
    running_mask = torch.cat(
        (attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)), dim=1
    )
    past = output.past_key_values if use_cache else None
    decode_ms: list[float] = []

    for _ in range(new_tokens - 1):
        model_input = next_token if use_cache else generated
        _sync(device)
        step_start = time.perf_counter()
        output = model(
            input_ids=model_input,
            attention_mask=running_mask,
            past_key_values=past,
            use_cache=use_cache,
        )
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        _sync(device)
        decode_ms.append((time.perf_counter() - step_start) * 1000.0)
        generated = torch.cat((generated, next_token), dim=1)
        running_mask = torch.cat(
            (running_mask, torch.ones_like(next_token, dtype=running_mask.dtype)), dim=1
        )
        past = output.past_key_values if use_cache else None

    end_to_end_ms = ttft_ms + sum(decode_ms)
    output_count = input_ids.shape[0] * new_tokens
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": sum(decode_ms) / len(decode_ms),
        "end_to_end_ms": end_to_end_ms,
        "output_tokens_per_second": output_count / (end_to_end_ms / 1000.0),
        "peak_memory_mib": peak_memory,
    }


def run(config: BenchmarkConfig, output_path: str | Path) -> dict:
    config.validate()
    torch.manual_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    dtype = _dtype(config.dtype) if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=dtype,
    ).to(device)
    model.eval()

    # Exercise both paths at the largest matrix point before measurement so
    # GPU power-state ramp and one-time allocator setup do not favor the
    # backend that happens to run second.
    warm_ids, warm_mask = _make_batch(
        tokenizer, max(config.context_lengths), max(config.batch_sizes), device
    )
    for use_cache in (True, False):
        _decode_once(model, warm_ids, warm_mask, min(config.new_tokens, 8), use_cache)
    del warm_ids, warm_mask

    points = [
        (context_length, batch_size)
        for context_length in config.context_lengths
        for batch_size in config.batch_sizes
    ]
    random.Random(config.seed).shuffle(points)

    experiments = []
    for context_length, batch_size in points:
        input_ids, attention_mask = _make_batch(
            tokenizer, context_length, batch_size, device
        )
        runs_by_backend = {backend: [] for backend in config.backends}
        for backend in config.backends:
            for _ in range(config.warmup_runs):
                _decode_once(
                    model,
                    input_ids,
                    attention_mask,
                    config.new_tokens,
                    backend == "transformers_kv_cache",
                )
        for repeat in range(config.measured_runs):
            order = list(config.backends)
            if repeat % 2:
                order.reverse()
            for backend in order:
                runs_by_backend[backend].append(
                    _decode_once(
                        model,
                        input_ids,
                        attention_mask,
                        config.new_tokens,
                        backend == "transformers_kv_cache",
                    )
                )
        for backend in config.backends:
            runs = runs_by_backend[backend]
            experiments.append(
                {
                    "backend": backend,
                    "context_length": context_length,
                    "batch_size": batch_size,
                    "runs": runs,
                    "summary": summarize_runs(runs),
                }
            )
    experiments.sort(key=lambda item: (item["backend"], item["context_length"], item["batch_size"]))

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": str(device),
            "gpu": gpu_name,
        },
        "measurement_scope": (
            "model prefill and autoregressive decode only; tokenization excluded"
        ),
        "experiment_order": (
            "seeded random matrix-point order; cached and uncached runs paired, "
            "with backend-first order alternating by repeat"
        ),
        "experiments": experiments,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
