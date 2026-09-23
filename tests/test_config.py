import json

import pytest

from inference_lab.config import BenchmarkConfig


def test_config_round_trip_and_validation(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model_id": "test/model",
                "device": "cpu",
                "dtype": "float32",
                "context_lengths": [8, 16],
                "batch_sizes": [1],
                "new_tokens": 4,
                "warmup_runs": 0,
                "measured_runs": 2,
                "seed": 1,
                "backends": ["transformers_kv_cache"],
            }
        ),
        encoding="utf-8",
    )
    config = BenchmarkConfig.from_json(path)
    config.validate()
    assert config.context_lengths == (8, 16)


def test_unknown_backend_is_rejected():
    config = BenchmarkConfig("x", "cpu", "float32", (8,), (1,), 2, 0, 1, 1, ("magic",))
    with pytest.raises(ValueError, match="unsupported backends"):
        config.validate()
