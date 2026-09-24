from fastapi.testclient import TestClient

from inference_lab.gateway_app import app


def test_gateway_health_metrics_and_overload_rejection() -> None:
    with TestClient(app) as client:
        health = client.get("/healthz")
        metrics = client.get("/metrics")
        rejected = client.post(
            "/v1/chat/completions",
            headers={"x-llm-deadline-ms": "1"},
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 64,
            },
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert "llm_gateway_queue_depth" in metrics.text
    assert rejected.status_code == 429
    assert rejected.json()["detail"]["reason"] == "predicted_slo_miss"
