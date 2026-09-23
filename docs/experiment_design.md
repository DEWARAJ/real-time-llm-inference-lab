# Experiment design

The benchmark separates four quantities that are often collapsed into one number:

- **Time to first token (TTFT):** prefill latency from model invocation to the first sampled token.
- **Time per output token (TPOT):** mean synchronized decode-step latency after prefill.
- **End-to-end latency:** TTFT plus every measured decode step.
- **Output throughput:** generated tokens across the batch divided by end-to-end time.

Tokenization is excluded and each CUDA timing boundary is synchronized. Every matrix point receives a warm-up run followed by repeated measured runs. The report retains raw samples and summarizes median and p95 values so that tail behavior remains visible.

Matrix points are visited in a seeded random order. At each point, cached and uncached measurements are paired, and the backend that runs first alternates on every repeat. This limits bias from laptop GPU power-state, clock, and thermal changes.

## Controlled comparison

The first experiment changes one inference decision: whether autoregressive decoding reuses the model's key/value cache. Model weights, input tokens, greedy decoding, context length, batch size, dtype, device, and output length remain fixed. The no-cache path recomputes the entire growing sequence at every decode step. This is deliberately inefficient; it supplies a transparent baseline that shows why cache layout and cache capacity matter.

The OpenAI-compatible server harness extends the same methodology to vLLM or SGLang. It records streaming TTFT, end-to-end request latency, concurrency, and aggregate output throughput without coupling the benchmark to a single server implementation.

## Limits

Results from a laptop RTX 4050 characterize this hardware and software configuration. They do not establish production-scale serving performance. The Transformers comparison isolates cache reuse but does not reproduce vLLM's paged attention, continuous batching, or scheduler. Those are measured separately through the server harness on a supported Linux environment.

The original matrix also included batch size 4. Full-sequence recomputation at 512 tokens exceeded the 6 GB device limit because the baseline materializes full-vocabulary logits for the growing sequence. The reported controlled matrix therefore uses batch sizes 1 and 2; the out-of-memory boundary is treated as a hardware limit rather than silently dropping a successful run.
