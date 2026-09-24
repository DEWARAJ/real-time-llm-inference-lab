from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = (
    "Explain how KV-cache layout, continuous batching, and memory bandwidth "
    "affect real-time language-model serving."
)
QUALITY_TEXT = " " .join(
    [
        "Real-time inference is a systems problem as much as a modeling problem.",
        "A useful benchmark separates prefill latency from autoregressive decode latency.",
        "Memory movement can dominate computation when tensors repeatedly cross device boundaries.",
        "Caching attention keys and values avoids recomputing the complete prefix at every step.",
        "Continuous batching improves throughput but can increase time to first token.",
        "Quantization reduces model memory and may trade arithmetic cost against output quality.",
        "Tail latency must be measured because a median can hide poor interactive behavior.",
        "Correctness checks belong beside performance measurements.",
    ]
    * 6
)


def distribution(samples: list[float]) -> dict[str, float]:
    return {
        "median": float(np.median(samples)),
        "p95": float(np.percentile(samples, 95)),
        "minimum": float(np.min(samples)),
        "maximum": float(np.max(samples)),
    }


def load_model(mode: str):
    if mode == "fp16":
        return AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float16, device_map={"": 0}
        )
    if mode == "nf4":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=quantization, device_map={"": 0}
        )
    raise ValueError(mode)


@torch.inference_mode()
def benchmark_mode(mode, tokenizer, warmup, repeats, new_tokens):
    torch.cuda.empty_cache()
    load_start = time.perf_counter()
    model = load_model(mode)
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    footprint_mib = model.get_memory_footprint() / 1024**2
    allocated_mib = torch.cuda.memory_allocated() / 1024**2

    prompt = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    generation_args = {
        **prompt,
        "max_new_tokens": new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    for _ in range(warmup):
        model.generate(**generation_args)
    generation_ms = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.generate(**generation_args)
        torch.cuda.synchronize()
        generation_ms.append((time.perf_counter() - start) * 1000.0)

    quality = tokenizer(
        QUALITY_TEXT,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to("cuda")
    loss = float(model(**quality, labels=quality["input_ids"]).loss.item())
    perplexity = math.exp(loss)
    latency = distribution(generation_ms)
    result = {
        "mode": mode,
        "load_seconds": load_seconds,
        "model_footprint_mib": footprint_mib,
        "cuda_allocated_after_load_mib": allocated_mib,
        "generation_ms": latency,
        "output_tokens_per_second": new_tokens / (latency["median"] / 1000.0),
        "quality_corpus_tokens": int(quality["input_ids"].numel()),
        "quality_cross_entropy": loss,
        "quality_perplexity": perplexity,
    }
    del quality, prompt, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--output", default="results/quantization.json")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    experiments = [
        benchmark_mode(mode, tokenizer, args.warmup, args.repeats, args.new_tokens)
        for mode in ("fp16", "nf4")
    ]
    fp16, nf4 = experiments
    report = {
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bitsandbytes": __import__("bitsandbytes").__version__,
        },
        "model": MODEL_ID,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "new_tokens": args.new_tokens,
        "experiments": experiments,
        "comparison": {
            "footprint_reduction_percent": 100.0
            * (1.0 - nf4["model_footprint_mib"] / fp16["model_footprint_mib"]),
            "throughput_change_percent": 100.0
            * (
                nf4["output_tokens_per_second"]
                / fp16["output_tokens_per_second"]
                - 1.0
            ),
            "perplexity_change_percent": 100.0
            * (nf4["quality_perplexity"] / fp16["quality_perplexity"] - 1.0),
        },
        "quality_scope": (
            "fixed 512-token engineering-text sanity corpus; not a general model-quality benchmark"
        ),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
