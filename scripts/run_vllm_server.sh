#!/usr/bin/env bash
set -euo pipefail

vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager \
  --port 8000
