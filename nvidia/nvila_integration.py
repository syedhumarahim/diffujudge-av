"""NVILA-8B integration — NVIDIA's open VLM family slotted into BaseJudge.

NVILA inherits VLLMJudge directly because it is supported by recent vLLM
releases via the standard chat-template route. The integration value here is
*signal*: using NVIDIA's own model in a research repo demonstrates fluency
with their stack.

Reference: NVILA, NVIDIA, 2024 — https://github.com/NVlabs/VILA
"""
from __future__ import annotations

from diffujudge.judges.vllm_judge import VLLMJudge


class NVILAJudge(VLLMJudge):
    DEFAULT_MODEL = "Efficient-Large-Model/NVILA-8B"

    def __init__(self, name: str = "nvila-8b", model: str | None = None, **kwargs) -> None:
        super().__init__(name=name, model=model or self.DEFAULT_MODEL, **kwargs)


def build_nvila_first_ensemble() -> list[VLLMJudge]:
    """An NVIDIA-flavored variant of the design's recommended ensemble.

    Replaces InternVL2-8B with NVILA-8B; keeps Qwen2.5-VL-7B as primary and
    LLaVA-Critic-7B as the specialized critic. Intended as an A/B against
    the canonical Qwen+InternVL+LLaVA-Critic ensemble.
    """
    return [
        VLLMJudge(name="qwen2.5-vl-7b", model="Qwen/Qwen2.5-VL-7B-Instruct"),
        NVILAJudge(),
        VLLMJudge(name="llava-critic-7b", model="lmms-lab/llava-critic-7b"),
    ]
