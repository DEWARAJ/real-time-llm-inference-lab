#!/usr/bin/env python3
"""Deterministic overload simulation for gateway policy evaluation.

This is a queueing simulation, not a GPU benchmark. Its purpose is to test
policy behavior before a live backend run and to keep simulated evidence
clearly separated from hardware measurements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference_lab.gateway import (  # noqa: E402
    AdmissionController,
    InferenceRequest,
    RequestQueue,
    SchedulingPolicy,
    percentile,
)


def make_trace(count: int, seed: int) -> list[InferenceRequest]:
    rng = random.Random(seed)
    now = 0.0
    trace: list[InferenceRequest] = []
    for index in range(count):
        # Alternate steady traffic with a burst every 100 requests.
        burst = index % 100 >= 70
        now += rng.expovariate(45.0 if burst else 12.0)
        interactive = rng.random() < 0.35
        prompt_tokens = rng.choice([64, 128, 256, 512, 1024])
        output_tokens = rng.choice([32, 64, 96])
        prefix_key = f"prefix-{rng.randrange(4)}" if rng.random() < 0.65 else None
        trace.append(
            InferenceRequest(
                request_id=f"request-{index:04d}",
                prompt_tokens=prompt_tokens,
                max_new_tokens=output_tokens,
                priority=10 if interactive else 0,
                prefix_key=prefix_key,
                deadline_ms=1_200.0 if interactive else 4_000.0,
                enqueued_at=now,
            )
        )
    return trace


def batch_service_seconds(batch: list[InferenceRequest]) -> float:
    batch_efficiency = 1.0 + 0.55 * (len(batch) - 1)
    prefill = max(item.prompt_tokens for item in batch) * 0.00008
    decode = sum(item.max_new_tokens for item in batch) / (350.0 * batch_efficiency)
    prefix_hits = sum(1 for item in batch if item.prefix_key == batch[0].prefix_key and item.prefix_key)
    prefix_factor = 0.92 if prefix_hits > 1 else 1.0
    return 0.035 + prefix_factor * (prefill + decode)


def simulate(
    trace: list[InferenceRequest],
    policy: SchedulingPolicy,
    admission_enabled: bool,
) -> dict[str, object]:
    queue = RequestQueue(policy)
    controller = AdmissionController(
        initial_service_tokens_per_second=350,
        base_ttft_ms=35,
        max_queue_requests=48,
        max_queue_tokens=24_576,
        safety_margin=1.05,
    )
    cursor = 0
    now = trace[0].enqueued_at if trace else 0.0
    busy_until: float | None = None
    inflight: list[InferenceRequest] = []
    accepted: dict[str, InferenceRequest] = {}
    rejected: dict[str, int] = {}
    latencies_ms: list[float] = []
    slo_met = 0
    class_counts = {"interactive": 0, "bulk": 0}
    class_slo_met = {"interactive": 0, "bulk": 0}
    queue_samples: list[int] = []

    while cursor < len(trace) or len(queue) or inflight:
        next_arrival = trace[cursor].enqueued_at if cursor < len(trace) else float("inf")
        next_completion = busy_until if busy_until is not None else float("inf")
        now = min(next_arrival, next_completion)

        if next_completion <= next_arrival:
            for request in inflight:
                latency = (now - request.enqueued_at) * 1_000
                latencies_ms.append(latency)
                slo_met += int(latency <= request.deadline_ms)
                request_class = "interactive" if request.priority > 0 else "bulk"
                class_counts[request_class] += 1
                class_slo_met[request_class] += int(latency <= request.deadline_ms)
            inflight = []
            busy_until = None

        while cursor < len(trace) and trace[cursor].enqueued_at <= now:
            request = trace[cursor]
            cursor += 1
            if admission_enabled:
                decision = controller.decide(request, queue)
                if not decision.accepted:
                    rejected[decision.reason] = rejected.get(decision.reason, 0) + 1
                    continue
            queue.push(request)
            accepted[request.request_id] = request

        if busy_until is None and len(queue):
            inflight = queue.pop_batch(max_requests=8, max_tokens=8_192, now=now)
            busy_until = now + batch_service_seconds(inflight)
        queue_samples.append(len(queue))

    duration = max(0.001, now - trace[0].enqueued_at) if trace else 0.001
    return {
        "policy": policy.value,
        "admission_control": admission_enabled,
        "requests_total": len(trace),
        "requests_accepted": len(accepted),
        "requests_rejected": sum(rejected.values()),
        "rejections_by_reason": rejected,
        "accepted_slo_rate_pct": 100 * slo_met / max(1, len(latencies_ms)),
        "accepted_slo_rate_by_class_pct": {
            key: 100 * class_slo_met[key] / max(1, class_counts[key]) for key in class_counts
        },
        "latency_ms": {
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
        },
        "completed_throughput_requests_per_second": len(latencies_ms) / duration,
        "mean_queue_depth": statistics.fmean(queue_samples) if queue_samples else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "gateway_simulation.json")
    args = parser.parse_args()
    trace = make_trace(args.requests, args.seed)
    runs = [
        simulate(trace, policy, admission)
        for policy in SchedulingPolicy
        for admission in (False, True)
    ]
    report = {
        "kind": "deterministic_queueing_simulation",
        "disclaimer": "These are simulated policy results, not measured GPU-serving results.",
        "seed": args.seed,
        "requests": args.requests,
        "model": {
            "output_tokens_per_second": 350,
            "dispatch_batch_size": 8,
            "dispatch_token_budget": 8192,
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
