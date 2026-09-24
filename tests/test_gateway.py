import pytest

from inference_lab.gateway import (
    AdmissionController,
    CircuitBreaker,
    CircuitState,
    GatewayMetrics,
    InferenceRequest,
    RequestQueue,
    SchedulingPolicy,
)


def request(
    request_id: str,
    *,
    prompt: int = 10,
    output: int = 10,
    enqueued: float = 0.0,
    priority: int = 0,
    prefix: str | None = None,
    deadline: float = 2_000,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        prompt_tokens=prompt,
        max_new_tokens=output,
        enqueued_at=enqueued,
        priority=priority,
        prefix_key=prefix,
        deadline_ms=deadline,
    )


def test_fifo_and_token_budget() -> None:
    queue = RequestQueue(SchedulingPolicy.FIFO)
    queue.push(request("one", enqueued=1))
    queue.push(request("two", enqueued=2, prompt=20))
    queue.push(request("three", enqueued=3))
    batch = queue.pop_batch(max_requests=3, max_tokens=45, now=4)
    assert [item.request_id for item in batch] == ["one", "three"]
    assert len(queue) == 1


def test_shortest_priority_and_prefix_policies() -> None:
    shortest = RequestQueue(SchedulingPolicy.SHORTEST_FIRST)
    shortest.push(request("large", prompt=100))
    shortest.push(request("small", prompt=5))
    assert shortest.pop_batch(max_requests=1, max_tokens=1_000)[0].request_id == "small"

    priority = RequestQueue(SchedulingPolicy.PRIORITY)
    priority.push(request("normal", priority=0))
    priority.push(request("interactive", priority=10))
    assert priority.pop_batch(max_requests=1, max_tokens=1_000)[0].request_id == "interactive"

    prefix = RequestQueue(SchedulingPolicy.PREFIX_AWARE)
    prefix.push(request("anchor", enqueued=0, prefix="shared"))
    prefix.push(request("other", enqueued=1, prefix="other"))
    prefix.push(request("matching", enqueued=2, prefix="shared"))
    assert [item.request_id for item in prefix.pop_batch(max_requests=2, max_tokens=1_000, now=3)] == [
        "anchor",
        "matching",
    ]


def test_admission_rejects_predicted_slo_miss_and_updates_rate() -> None:
    queue = RequestQueue()
    controller = AdmissionController(initial_service_tokens_per_second=100, safety_margin=1.0)
    decision = controller.decide(request("slow", prompt=100, output=100, deadline=500), queue)
    assert not decision.accepted
    assert decision.reason == "predicted_slo_miss"
    controller.observe(completed_tokens=200, elapsed_seconds=1)
    assert controller.service_tokens_per_second == pytest.approx(120)


def test_circuit_breaker_recovers_through_half_open() -> None:
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=5)
    breaker.record_failure(now=10)
    assert breaker.allow(now=11)
    breaker.record_failure(now=12)
    assert breaker.state is CircuitState.OPEN
    assert not breaker.allow(now=16)
    assert breaker.allow(now=17)
    assert breaker.state is CircuitState.HALF_OPEN
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


def test_prometheus_metrics_include_rejection_reason() -> None:
    queue = RequestQueue()
    controller = AdmissionController(initial_service_tokens_per_second=1)
    decision = controller.decide(request("miss", deadline=1), queue)
    metrics = GatewayMetrics()
    metrics.record_decision(decision)
    metrics.observe_ttft(42.5)
    rendered = metrics.render()
    assert 'reason="predicted_slo_miss"' in rendered
    assert "llm_gateway_ttft_ms_count 1" in rendered
