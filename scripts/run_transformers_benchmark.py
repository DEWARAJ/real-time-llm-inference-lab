from __future__ import annotations

import argparse
import json

from inference_lab.config import BenchmarkConfig
from inference_lab.transformers_benchmark import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rtx4050_qwen.json")
    parser.add_argument("--output", default="results/transformers_benchmark.json")
    args = parser.parse_args()
    report = run(BenchmarkConfig.from_json(args.config), args.output)
    print(json.dumps({"output": args.output, "experiments": len(report["experiments"])}, indent=2))


if __name__ == "__main__":
    main()
