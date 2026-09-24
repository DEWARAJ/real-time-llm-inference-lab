"""SLO-aware admission control and scheduling for LLM inference requests.

The core is dependency-free so the scheduling and overload behavior can be
tested independently of a web framework or model server.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Iterable


class SchedulingPolicy(str, Enum):
    FIFO = "fifo"
    SHORTEST_FIRST = "shortest_first"
    PRIORITY = "priority"
    PREFIX_AWARE = "prefix_aware"


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    enqueued_at: float
    priority: int = 0
    prefix_key: str | None = None
    deadline_ms: float = 1_000.0

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.max_new_tokens <= 0:
            raise ValueError("token counts must be non-negative and output must be positive")
        if self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.max_new_tokens


class RequestQueue:
    """In-memory policy queue with token-budgeted batch selection."""

    def __init__(self, policy: SchedulingPolicy = SchedulingPolicy.FIFO) -> None:
        self.policy = SchedulingPolicy(policy)
        self._requests: list[InferenceRequest] = []

    def __len__(self) -> int:
        return len(self._requests)

    @property
    def queued_tokens(self) -> int:
        return sum(request.total_tokens for request in self._requests)

    def push(self, request: InferenceRequest) -> None:
        if any(item.request_id == request.request_id for item in self._requests):
            raise ValueError(f"duplicate request_id: {request.request_id}")
        self._requests.append(request)

    def remove(self, request_id: str) -> InferenceRequest | None:
        for index, request in enumerate(self._requests):
            if request.request_id == request_id:
                return self._requests.pop(index)
        return None

    def _ordered(self, now: float) -> list[InferenceRequest]:
        if self.policy is SchedulingPolicy.FIFO:
            return sorted(self._requests, key=lambda item: (item.enqueued_at, item.request_id))
        if self.policy is SchedulingPolicy.SHORTEST_FIRST:
            return sorted(
                self._requests,
                key=lambda item: (item.total_tokens, item.enqueued_at, item.request_id),
            )
        if self.policy is SchedulingPolicy.PRIORITY:
            return sorted(
                self._requests,
                key=lambda item: (
                    -item.priority,
                    item.enqueued_at + item.deadline_ms / 1_000,
                    item.enqueued_at,
                ),
            )

        # Prefix-aware scheduling chooses the oldest request as an anchor, then
        # co-schedules matching prefixes before unrelated work.
        anchor = min(self._requests, key=lambda item: (item.enqueued_at, item.request_id))
        return sorted(
            self._requests,
            key=lambda item: (
                0 if anchor.prefix_key and item.prefix_key == anchor.prefix_key else 1,
                max(0.0, now - item.enqueued_at) * -1,
                item.enqueued_at,
            ),
        )

    def pop_batch(
        self,
        *,
        max_requests: int,
        max_tokens: int,
        now: float | None = None,
    ) -> list[InferenceRequest]:
        if max_requests <= 0 or max_tokens <= 0:
            raise ValueError("batch limits must be positive")
        now = time.monotonic() if now is None else now
        selected: list[InferenceRequest] = []
        selected_tokens = 0
        for request in self._ordered(now):
            if len(selected) >= max_requests:
                break
            if selected and selected_tokens + request.total_tokens > max_tokens:
                continue
            if not selected and request.total_tokens > max_tokens:
                # Always make progress; an oversized request is handled alone.
                selected.append(request)
                break
            selected.append(request)
            selected_tokens += request.total_tokens
        selected_ids = {request.request_id for request in selected}
        self._requests = [item for item in self._requests if item.request_id not in selected_ids]
        return selected


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    accepted: bool
    reason: str
    predicted_completion_ms: float


class AdmissionController:
    """Predicts queue completion time and rejects work that cannot meet its SLO."""

    def __init__(
        self,
        *,
        initial_service_tokens_per_second: float = 40.0,
        base_ttft_ms: float = 50.0,
        max_queue_requests: int = 64,
        max_queue_tokens: int = 32_768,
        safety_margin: float = 1.15,
        ewma_alpha: float = 0.2,
    ) -> None:
        if initial_service_tokens_per_second <= 0:
            raise ValueError("service rate must be positive")
        if not 0 < ewma_alpha <= 1:
            raise ValueError("ewma_alpha must be in (0, 1]")
        self.service_tokens_per_second = initial_service_tokens_per_second
        self.base_ttft_ms = base_ttft_ms
        self.max_queue_requests = max_queue_requests
        self.max_queue_tokens = max_queue_tokens
        self.safety_margin = safety_margin
        self.ewma_alpha = ewma_alpha

    def predict_completion_ms(self, queued_tokens: int, request_tokens: int) -> float:
        service_ms = 1_000 * (queued_tokens + request_tokens) / self.service_tokens_per_second
        return self.base_ttft_ms + self.safety_margin * service_ms

    def decide(self, request: InferenceRequest, queue: RequestQueue) -> AdmissionDecision:
        predicted = self.predict_completion_ms(queue.queued_tokens, request.total_tokens)
        if len(queue) >= self.max_queue_requests:
            return AdmissionDecision(False, "queue_request_limit", predicted)
        if queue.queued_tokens + request.total_tokens > self.max_queue_tokens:
            return AdmissionDecision(False, "queue_token_limit", predicted)
        if predicted > request.deadline_ms:
            return AdmissionDecision(False, "predicted_slo_miss", predicted)
        return AdmissionDecision(True, "accepted", predicted)

    def observe(self, *, completed_tokens: int, elapsed_seconds: float) -> None:
        if completed_tokens <= 0 or elapsed_seconds <= 0:
            return
        observed = completed_tokens / elapsed_seconds
        alpha = self.ewma_alpha
        self.service_tokens_per_second = alpha * observed + (1 - alpha) * self.service_tokens_per_second


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_seconds: float = 10.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.failures = 0
        self.state = CircuitState.CLOSED
        self.opened_at: float | None = None

    def allow(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if self.state is CircuitState.OPEN and self.opened_at is not None:
            if now - self.opened_at >= self.recovery_seconds:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.state = CircuitState.CLOSED
        self.opened_at = None

    def record_failure(self, now: float | None = None) -> None:
        self.failures += 1
        if self.state is CircuitState.HALF_OPEN or self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic() if now is None else now


class GatewayMetrics:
    """Small Prometheus text exporter without a runtime dependency."""

    def __init__(self) -> None:
        self.accepted = 0
        self.rejected: dict[str, int] = {}
        self.completed = 0
        self.failed = 0
        self.queue_depth = 0
        self.queue_tokens = 0
        self.ttft_ms: list[float] = []

    def record_decision(self, decision: AdmissionDecision) -> None:
        if decision.accepted:
            self.accepted += 1
        else:
            self.rejected[decision.reason] = self.rejected.get(decision.reason, 0) + 1

    def observe_ttft(self, value_ms: float) -> None:
        if math.isfinite(value_ms) and value_ms >= 0:
            self.ttft_ms.append(value_ms)

    def render(self) -> str:
        lines = [
            "# HELP llm_gateway_requests_accepted_total Accepted requests.",
            "# TYPE llm_gateway_requests_accepted_total counter",
            f"llm_gateway_requests_accepted_total {self.accepted}",
            "# HELP llm_gateway_requests_rejected_total Rejected requests by reason.",
            "# TYPE llm_gateway_requests_rejected_total counter",
        ]
        for reason, count in sorted(self.rejected.items()):
            lines.append(f'llm_gateway_requests_rejected_total{{reason="{reason}"}} {count}')
        lines.extend(
            [
                "# TYPE llm_gateway_queue_depth gauge",
                f"llm_gateway_queue_depth {self.queue_depth}",
                "# TYPE llm_gateway_queue_tokens gauge",
                f"llm_gateway_queue_tokens {self.queue_tokens}",
                "# TYPE llm_gateway_requests_completed_total counter",
                f"llm_gateway_requests_completed_total {self.completed}",
                "# TYPE llm_gateway_requests_failed_total counter",
                f"llm_gateway_requests_failed_total {self.failed}",
            ]
        )
        if self.ttft_ms:
            lines.extend(
                [
                    "# TYPE llm_gateway_ttft_ms summary",
                    f"llm_gateway_ttft_ms_count {len(self.ttft_ms)}",
                    f"llm_gateway_ttft_ms_sum {sum(self.ttft_ms):.6f}",
                ]
            )
        return "\n".join(lines) + "\n"


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[max(index, 0)]
