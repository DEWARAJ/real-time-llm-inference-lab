"""FastAPI gateway for an OpenAI-compatible vLLM or SGLang backend."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import os
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .gateway import (
    AdmissionController,
    CircuitBreaker,
    GatewayMetrics,
    InferenceRequest,
    RequestQueue,
    SchedulingPolicy,
)


def _estimate_tokens(payload: dict[str, Any]) -> tuple[int, int, str | None]:
    messages = payload.get("messages", [])
    prompt = "\n".join(str(item.get("content", "")) for item in messages)
    prompt_tokens = max(1, len(prompt) // 4)
    max_new_tokens = int(payload.get("max_tokens", 64))
    prefix = prompt[:256].encode("utf-8")
    prefix_key = hashlib.sha256(prefix).hexdigest()[:16] if prefix else None
    return prompt_tokens, max_new_tokens, prefix_key


@dataclass(slots=True)
class PendingRequest:
    request: InferenceRequest
    dispatch: asyncio.Future[None]


class GatewayRuntime:
    def __init__(
        self,
        *,
        policy: SchedulingPolicy,
        batch_size: int,
        batch_tokens: int,
        scheduling_window_ms: float,
    ) -> None:
        self.queue = RequestQueue(policy)
        self.controller = AdmissionController(
            initial_service_tokens_per_second=float(os.getenv("SERVICE_TOKENS_PER_SECOND", "40")),
            max_queue_requests=int(os.getenv("MAX_QUEUE_REQUESTS", "64")),
            max_queue_tokens=int(os.getenv("MAX_QUEUE_TOKENS", "32768")),
        )
        self.breaker = CircuitBreaker()
        self.metrics = GatewayMetrics()
        self.batch_size = batch_size
        self.batch_tokens = batch_tokens
        self.scheduling_window_ms = scheduling_window_ms
        self._pending: dict[str, PendingRequest] = {}
        self._condition = asyncio.Condition()
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def admit(self, request: InferenceRequest) -> tuple[bool, str, float]:
        if not self.breaker.allow():
            return False, "backend_circuit_open", 0.0
        async with self._condition:
            decision = self.controller.decide(request, self.queue)
            self.metrics.record_decision(decision)
            if not decision.accepted:
                return False, decision.reason, decision.predicted_completion_ms
            loop = asyncio.get_running_loop()
            pending = PendingRequest(request, loop.create_future())
            self._pending[request.request_id] = pending
            self.queue.push(request)
            self._update_queue_metrics()
            self._condition.notify()
        await pending.dispatch
        return True, "accepted", decision.predicted_completion_ms

    def _update_queue_metrics(self) -> None:
        self.metrics.queue_depth = len(self.queue)
        self.metrics.queue_tokens = self.queue.queued_tokens

    async def _dispatch_loop(self) -> None:
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: len(self.queue) > 0)
            await asyncio.sleep(self.scheduling_window_ms / 1_000)
            async with self._condition:
                batch = self.queue.pop_batch(
                    max_requests=self.batch_size,
                    max_tokens=self.batch_tokens,
                )
                self._update_queue_metrics()
            for request in batch:
                pending = self._pending.pop(request.request_id, None)
                if pending and not pending.dispatch.done():
                    pending.dispatch.set_result(None)


def create_app():
    backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
    runtime = GatewayRuntime(
        policy=SchedulingPolicy(os.getenv("SCHEDULING_POLICY", "priority")),
        batch_size=int(os.getenv("DISPATCH_BATCH_SIZE", "8")),
        batch_tokens=int(os.getenv("DISPATCH_BATCH_TOKENS", "8192")),
        scheduling_window_ms=float(os.getenv("SCHEDULING_WINDOW_MS", "5")),
    )
    app = FastAPI(title="SLO-Aware LLM Inference Gateway", version="0.1.0")

    @app.on_event("startup")
    async def startup() -> None:
        await runtime.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await runtime.stop()

    @app.get("/healthz")
    async def healthz():
        status = "ok" if runtime.breaker.allow() else "degraded"
        return {"status": status, "queue_depth": len(runtime.queue)}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return runtime.metrics.render()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        prompt_tokens, max_new_tokens, prefix_key = _estimate_tokens(payload)
        deadline_ms = float(request.headers.get("x-llm-deadline-ms", "2000"))
        priority = int(request.headers.get("x-llm-priority", "0"))
        ticket = InferenceRequest(
            request_id=str(uuid.uuid4()),
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            priority=priority,
            prefix_key=prefix_key,
            deadline_ms=deadline_ms,
            enqueued_at=time.monotonic(),
        )
        accepted, reason, predicted_ms = await runtime.admit(ticket)
        if not accepted:
            raise HTTPException(
                status_code=429 if reason != "backend_circuit_open" else 503,
                detail={"reason": reason, "predicted_completion_ms": predicted_ms},
            )

        target = f"{backend_url}/v1/chat/completions"
        started = time.perf_counter()
        if payload.get("stream"):
            async def stream_backend():
                try:
                    async with httpx.AsyncClient(timeout=None) as client:
                        async with client.stream("POST", target, json=payload) as response:
                            response.raise_for_status()
                            first = True
                            async for chunk in response.aiter_bytes():
                                if first:
                                    runtime.metrics.observe_ttft((time.perf_counter() - started) * 1_000)
                                    first = False
                                yield chunk
                    runtime.breaker.record_success()
                    runtime.metrics.completed += 1
                    runtime.controller.observe(
                        completed_tokens=ticket.total_tokens,
                        elapsed_seconds=max(time.perf_counter() - started, 1e-6),
                    )
                except Exception:
                    runtime.breaker.record_failure()
                    runtime.metrics.failed += 1
                    raise

            return StreamingResponse(stream_backend(), media_type="text/event-stream")

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(target, json=payload)
                response.raise_for_status()
            runtime.metrics.observe_ttft((time.perf_counter() - started) * 1_000)
            runtime.metrics.completed += 1
            runtime.breaker.record_success()
            runtime.controller.observe(
                completed_tokens=ticket.total_tokens,
                elapsed_seconds=max(time.perf_counter() - started, 1e-6),
            )
            return JSONResponse(response.json(), status_code=response.status_code)
        except httpx.HTTPError as exc:
            runtime.metrics.failed += 1
            runtime.breaker.record_failure()
            raise HTTPException(status_code=502, detail=f"backend request failed: {exc}") from exc

    app.state.gateway_runtime = runtime
    return app


app = create_app()
