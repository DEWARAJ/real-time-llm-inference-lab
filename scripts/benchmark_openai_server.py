from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

from inference_lab.metrics import summarize


async def request_once(session, url, model, prompt, max_tokens, semaphore):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with semaphore:
        start = time.perf_counter()
        first_token = None
        completion_tokens = 0
        async with session.post(f"{url.rstrip('/')}/v1/completions", json=payload) as response:
            response.raise_for_status()
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                text = event.get("choices", [{}])[0].get("text", "") if event.get("choices") else ""
                if text and first_token is None:
                    first_token = time.perf_counter()
                usage = event.get("usage") or {}
                completion_tokens = max(completion_tokens, usage.get("completion_tokens", 0))
        end = time.perf_counter()
    ttft = ((first_token or end) - start) * 1000.0
    total = (end - start) * 1000.0
    return {"ttft_ms": ttft, "end_to_end_ms": total, "completion_tokens": completion_tokens}


async def run(args):
    prompt = "Explain why measuring memory movement matters in real-time inference. " * args.prompt_repetitions
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _ in range(args.warmup_requests):
            await request_once(
                session, args.url, args.model, prompt, args.max_tokens, asyncio.Semaphore(1)
            )
        tasks = [
            request_once(session, args.url, args.model, prompt, args.max_tokens, semaphore)
            for _ in range(args.requests)
        ]
        wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_seconds = time.perf_counter() - wall_start
    total_tokens = sum(item["completion_tokens"] for item in results)
    report = {
        "server": args.url,
        "model": args.model,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "warmup_requests": args.warmup_requests,
        "ttft_ms": summarize(item["ttft_ms"] for item in results),
        "end_to_end_ms": summarize(item["end_to_end_ms"] for item in results),
        "aggregate_output_tokens_per_second": total_tokens / wall_seconds if total_tokens else None,
        "raw": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Benchmark a vLLM/SGLang OpenAI-compatible server")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-repetitions", type=int, default=8)
    parser.add_argument("--output", default="results/server_benchmark.json")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
