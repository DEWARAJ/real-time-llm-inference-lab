# Resume bullets

**Real-Time LLM Inference Optimization Lab** | PyTorch, CUDA, vLLM, Triton, FlashAttention

- Built a reproducible Qwen2.5 inference benchmark measuring TTFT, TPOT, p50/p95 latency, throughput, and peak GPU memory; KV-cache reuse cut median decode latency 64.6% and p95 end-to-end latency 58.8% at 512-token context and batch size 2.
- Deployed an OpenAI-compatible vLLM server with FlashAttention and continuous batching, scaling aggregate throughput from 50.2 to 338.0 tokens/s across 1–8 concurrent requests while keeping median streaming TTFT below 88 ms.
- Implemented a fused FP16 Triton RMSNorm kernel with PyTorch reference validation, delivering 2.83–6.06× median speedup across 100 post-warm-up runs with maximum absolute error of 0.0078125.

Use the first two bullets when space is limited. Add the Triton bullet for inference or ML-systems roles.
