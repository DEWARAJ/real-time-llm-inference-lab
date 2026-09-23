import pytest

from inference_lab.metrics import percentile, summarize


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)


def test_summary_contains_tail_latency():
    report = summarize([10, 20, 30])
    assert report["median"] == 20
    assert report["p95"] == pytest.approx(29)
