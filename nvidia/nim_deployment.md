# Deploying the DiffuJudge-AV ensemble as an NVIDIA NIM endpoint

NVIDIA Inference Microservices (NIMs) expose Hugging Face VLMs behind an
OpenAI-compatible REST API. The judge layer here is built specifically so it
can be flipped from in-process vLLM to a remote NIM endpoint by changing one
constructor argument:

```python
from diffujudge.judges.vllm_judge import VLLMJudge

qwen = VLLMJudge(
    name="qwen2.5-vl-7b",
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    server_url="http://nim-qwen25-vl-7b.svc.cluster.local",
)
```

The judge will issue OpenAI-Chat-Completions requests; vLLM and NIM both
honor that wire format.

## Recommended deployment shape

```
                 ┌──────────────────────────┐
                 │   diffujudge runner      │
                 │ (Tweedie + Conformal)    │
                 └────────┬─────────────────┘
                          │  HTTP (OpenAI Chat Completions)
       ┌──────────────────┼──────────────────┐
       │                  │                  │
       ▼                  ▼                  ▼
   NIM-Qwen2.5-VL    NIM-NVILA-8B    NIM-LLaVA-Critic
```

Each NIM is a single-VLM service so they can scale independently — Qwen2.5-VL
is the heaviest because it does dynamic-FPS video, so target 1× H100 per
replica; the other two fit on L40S.

## Cost & latency

For the design's 200-clip × 7-level × 3-sample × 3-judge workload, the steady
state is ~31 500 forward passes. With three replicas, target throughput
≥ 60 req/s aggregate; total wall clock ≈ 9 minutes. This is roughly 25× the
sequential single-GPU number in §5.3 of the design — appropriate for a
production AV Eval scenario where ~100 K clips/day need scoring.

## Authentication

NIMs use the standard NVIDIA API key header. Set `NVIDIA_API_KEY` in `.env`
and `LiteLLMJudge` will pass it through automatically; for `VLLMJudge`, set
`server_url` and the runner will route through the proxy with no further
config.
