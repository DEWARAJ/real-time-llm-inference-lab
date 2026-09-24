# Benchmark report

## Environment

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB
- Local benchmark: PyTorch 2.9.1+cu128, Transformers 4.57.6, Windows
- Serving and kernel benchmark: vLLM 0.30.0, PyTorch 2.13.0+cu130, CUDA 13.0, Ubuntu 22.04 under WSL2
- Model: Qwen/Qwen2.5-0.5B-Instruct, FP16

## 1. Controlled KV-cache experiment

The local benchmark compares two greedy autoregressive decoding paths with identical weights and tokens. The cached path passes prior key/value tensors into each decode step. The baseline recomputes the full growing sequence.

The six matrix points cover context lengths 64, 256, and 512 with batch sizes 1 and 2. Each point receives two warm-up runs followed by four measured runs per backend. Cached and uncached measurements are paired, and the backend-first order alternates to reduce clock and thermal bias.

The largest point showed a 64.6% TPOT reduction, 169% output-throughput increase, 58.8% p95 end-to-end reduction, and 19.5% peak-memory reduction. The original matrix included batch size 4, but the full-recomputation baseline exceeded the 6 GB device limit at 512 tokens because it materializes full-vocabulary logits for the complete growing sequence. That boundary is documented rather than discarded.

Raw report: [`results/transformers_benchmark.json`](../results/transformers_benchmark.json)

## 2. vLLM serving experiment

The server was launched with:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager \
  --port 8000
```

The engine selected FlashAttention, enabled prefix caching and chunked prefill, and allocated a 3.22 GiB KV cache with capacity for 280,976 tokens. The load generator sent streaming OpenAI-compatible completion requests after two warm-up requests. Every request used the same prompt and generated 64 tokens, representing a shared-prefix serving workload.

Measured throughput rose from 50.24 tokens/s at concurrency 1 to 338.03 tokens/s at concurrency 8, a 6.73× increase. Median TTFT rose from 44.57 ms to 87.40 ms, exposing the latency-throughput tradeoff created by batching.

Raw reports:

- [`results/vllm_concurrency_1.json`](../results/vllm_concurrency_1.json)
- [`results/vllm_concurrency_4.json`](../results/vllm_concurrency_4.json)
- [`results/vllm_concurrency_8.json`](../results/vllm_concurrency_8.json)

## 3. Triton RMSNorm

The fused kernel computes variance, reciprocal RMS, normalization, and learned scaling in one Triton program. The benchmark checks its output against a float-accumulating PyTorch reference and fails if maximum absolute error exceeds 1e-2.

Across three representative shapes and 100 timed runs, the kernel produced 2.83–6.06× median speedup with maximum absolute error 0.0078125.

Raw report: [`results/triton_rmsnorm.json`](../results/triton_rmsnorm.json)

## 4. NF4 quantization

The quantization experiment loads the same model in FP16 and BitsAndBytes NF4 with double quantization and FP16 compute. It measures model footprint, CUDA allocation after load, 64-token greedy generation, and cross-entropy on a fixed 512-token engineering-text corpus.

NF4 reduced the model footprint from 942.3 MiB to 430.4 MiB, a 54.3% reduction. Median generation throughput fell from 37.16 to 23.44 tokens/s, a 36.9% regression, and sanity-corpus perplexity increased from 2.851 to 2.971, or 4.2%. The experiment demonstrates that low-bit weight storage is a capacity optimization on this hardware and workload, not automatically a latency optimization.

Raw report: [`results/quantization.json`](../results/quantization.json)

## Limits

These results characterize one laptop GPU and are not production capacity claims. Eager mode was used because it reduced startup complexity on the 6 GB device; CUDA graphs could change both latency and memory. The vLLM workload uses a shared prefix, so results should not be generalized to unrelated prompts without a separate experiment. HTTP results include tokenization and transport; the local model comparison excludes tokenization. The quantization quality corpus is a deterministic sanity check rather than a general language-model evaluation.
