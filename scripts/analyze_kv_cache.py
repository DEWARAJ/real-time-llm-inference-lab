from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig

from inference_lab.kv_cache import kv_cache_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lengths", nargs="+", type=int, default=[512, 2048, 8192, 32768])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output", default="results/kv_cache_analysis.json")
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model)
    head_dim = config.hidden_size // config.num_attention_heads
    rows = []
    for batch in args.batches:
        for length in args.lengths:
            byte_count = kv_cache_bytes(
                layers=config.num_hidden_layers,
                kv_heads=config.num_key_value_heads,
                head_dim=head_dim,
                sequence_length=length,
                batch_size=batch,
                dtype=args.dtype,
            )
            rows.append({"batch_size": batch, "sequence_length": length, "mib": byte_count / 1024**2})
    report = {
        "model": args.model,
        "layers": config.num_hidden_layers,
        "attention_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "head_dim": head_dim,
        "dtype": args.dtype,
        "rows": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
