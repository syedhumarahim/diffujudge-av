"""vLLM-backed VLM judges: Qwen2.5-VL-7B / InternVL2-8B / LLaVA-Critic-7B.

Per the design's §5.2 stack table, vLLM is the inference layer for the open
VLM ensemble. This module provides a thin wrapper that:

  • lazy-imports vllm so the rest of the package runs on CPU-only machines,
  • renders a model-specific multimodal prompt (Qwen2.5-VL uses dynamic FPS
    + MRoPE temporal encoding; LLaVA-Critic and InternVL2 use frame-list
    chat templates),
  • parses a structured JSON score reply,
  • supports both in-process LLM and an OpenAI-compatible vLLM server URL.

If `vllm` is not installed, instantiation raises a clear ImportError with the
correct extra to install. CI does not exercise this module — see
tests/test_judges_smoke.py for the contract test that validates the ABC.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from diffujudge.judges.api_judge import _build_prompt, parse_score_from_text
from diffujudge.judges.base import BaseJudge, JudgeOutput, RubricRequest


_FAMILY_TEMPLATES = {
    "qwen2.5-vl": "qwen_vl",
    "internvl2": "internvl_chat",
    "llava-critic": "llava_v1",
    "nvila": "nvila",
}


def _detect_family(model: str) -> str:
    m = model.lower()
    for key in _FAMILY_TEMPLATES:
        if key in m:
            return key
    return "generic"


class VLLMJudge(BaseJudge):
    """Single-model vLLM judge. Use one instance per VLM in the ensemble."""

    def __init__(
        self,
        name: str,
        model: str,
        score_min: float = 1.0,
        score_max: float = 5.0,
        temperature_default: float = 0.0,
        max_tokens: int = 512,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        server_url: str | None = None,
    ) -> None:
        super().__init__(name=name, score_min=score_min, score_max=score_max)
        self.model = model
        self.family = _detect_family(model)
        self.temperature_default = float(temperature_default)
        self.max_tokens = int(max_tokens)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.max_model_len = int(max_model_len)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.server_url = server_url or os.getenv("DIFFUJUDGE_VLLM_URL")
        self._llm: Any = None  # lazy

    def _ensure_llm(self) -> None:
        if self._llm is not None or self.server_url:
            return
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "VLLMJudge requires `pip install diffujudge-av[vllm]` and a CUDA GPU."
            ) from e
        self._llm = LLM(
            model=self.model,
            trust_remote_code=True,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt={"image": 16, "video": 1},
        )

    def _load_frames(self, paths: list[str]) -> list[Any]:
        from PIL import Image  # noqa: WPS433 — lazy
        return [Image.open(Path(p)).convert("RGB") for p in paths]

    def _format_messages(self, request: RubricRequest) -> list[dict[str, Any]]:
        view = request.view
        text = _build_prompt(request)
        if not view.frames:
            return [{"role": "user", "content": text}]
        if self.family in {"qwen2.5-vl", "qwen_vl"}:
            content: list[dict[str, Any]] = [
                {"type": "video", "video": [str(p) for p in view.frames]},
                {"type": "text", "text": text},
            ]
        else:
            content = [{"type": "image", "image": str(p)} for p in view.frames]
            content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def judge_one(self, request: RubricRequest) -> JudgeOutput:
        if self.server_url:
            return self._judge_via_server(request)
        return self._judge_in_process(request)

    def judge_batch(self, requests: list[RubricRequest]) -> list[JudgeOutput]:
        if self.server_url:
            return [self.judge_one(r) for r in requests]
        self._ensure_llm()
        from vllm import SamplingParams

        prompts = [self._format_messages(r) for r in requests]
        sps = [
            SamplingParams(
                temperature=r.view.temperature or self.temperature_default,
                top_p=1.0,
                max_tokens=self.max_tokens,
            )
            for r in requests
        ]
        t0 = time.perf_counter()
        outs = self._llm.chat(prompts, sampling_params=sps)
        dt = (time.perf_counter() - t0) / max(len(requests), 1)
        results: list[JudgeOutput] = []
        for r, o in zip(requests, outs, strict=True):
            text = o.outputs[0].text if o.outputs else ""
            score = self.clip(parse_score_from_text(text, fmt=r.view.score_id_format))
            results.append(
                JudgeOutput(
                    item_id=r.view.item_id,
                    sample_id=r.view.meta.get("sample_id", r.view.item_id),
                    judge_name=self.name,
                    score=score,
                    rationale=text[:500],
                    raw_response=text,
                    cost_usd=0.0,
                    latency_s=dt,
                    meta={"model": self.model, "family": self.family},
                )
            )
        return results

    def _judge_in_process(self, request: RubricRequest) -> JudgeOutput:
        return self.judge_batch([request])[0]

    def _judge_via_server(self, request: RubricRequest) -> JudgeOutput:
        import httpx  # noqa: WPS433

        t0 = time.perf_counter()
        view = request.view
        prompt = _build_prompt(request)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": view.temperature or self.temperature_default,
            "max_tokens": self.max_tokens,
        }
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(f"{self.server_url}/v1/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            return JudgeOutput(
                item_id=view.item_id,
                sample_id=view.meta.get("sample_id", view.item_id),
                judge_name=self.name,
                score=(self.score_min + self.score_max) / 2,
                rationale=f"(vllm-server error) {type(e).__name__}: {e}",
                latency_s=time.perf_counter() - t0,
                meta={"error": True},
            )
        score = self.clip(parse_score_from_text(text, fmt=view.score_id_format))
        return JudgeOutput(
            item_id=view.item_id,
            sample_id=view.meta.get("sample_id", view.item_id),
            judge_name=self.name,
            score=score,
            rationale=text[:500],
            raw_response=text,
            latency_s=time.perf_counter() - t0,
            meta={"model": self.model, "family": self.family, "server": True},
        )
