"""Anthropic-SDK-backed judge with parallel batching.

Uses the official `anthropic` Python SDK and supports the recommended ensemble
diversity pattern: same provider, different model+temperature so we get
genuine inter-judge variance with a single API key.

Recommended ensemble (1 key, 3 judges):
    - claude-haiku-4-5  @ T=0.0   (primary, cheap)
    - claude-sonnet-4-5 @ T=0.0   (heavier, diverse architecture-tier)
    - claude-haiku-4-5  @ T=0.6   (warm variant — sampling-temp diversity)

This gives ~3× cost of one Haiku call (Sonnet ≈ 5× Haiku, two Haiku ≈ 2×) at
~3× the diversity. For 200 items × 22 perturbed views = 4 400 calls per
judge → ~13 000 total calls, ~$3–8 budget at current pricing.

Prompt cache: enabled by default for the system prompt + rubric — saves
~70% on repeated prompts in the perturbation cascade.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from diffujudge.judges.api_judge import _build_prompt, parse_score_from_text
from diffujudge.judges.base import BaseJudge, JudgeOutput, RubricRequest


_SYSTEM = (
    "You are an expert evaluator for autonomous-driving question answering. "
    "You score model answers on a strict ordinal rubric and return JSON only."
)


class AnthropicJudge(BaseJudge):
    def __init__(
        self,
        name: str,
        model: str = "claude-haiku-4-5",
        score_min: float = 1.0,
        score_max: float = 5.0,
        temperature_default: float = 0.0,
        max_tokens: int = 256,
        enable_caching: bool = True,
        max_concurrency: int = 12,
    ) -> None:
        super().__init__(name=name, score_min=score_min, score_max=score_max)
        self.model = model
        self.temperature_default = float(temperature_default)
        self.max_tokens = int(max_tokens)
        self.enable_caching = bool(enable_caching)
        self.max_concurrency = int(max_concurrency)
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("AnthropicJudge requires `pip install anthropic`") from e
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise OSError(
                "ANTHROPIC_API_KEY not set. Did you load .env with override=True?"
            )
        self._client = anthropic.Anthropic()

    def judge_one(self, request: RubricRequest) -> JudgeOutput:
        t0 = time.perf_counter()
        view = request.view
        prompt = _build_prompt(request)

        system_block: list[dict[str, Any]] = [
            {"type": "text", "text": _SYSTEM},
        ]
        if self.enable_caching:
            system_block[0]["cache_control"] = {"type": "ephemeral"}

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": view.temperature or self.temperature_default,
            "system": system_block,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            resp = self._client.messages.create(**kwargs)
            text = resp.content[0].text if resp.content else ""
            usage = resp.usage
            cost = _estimate_cost(self.model, usage.input_tokens, usage.output_tokens,
                                  getattr(usage, "cache_read_input_tokens", 0) or 0,
                                  getattr(usage, "cache_creation_input_tokens", 0) or 0)
        except Exception as e:  # noqa: BLE001
            return JudgeOutput(
                item_id=view.item_id,
                sample_id=view.meta.get("sample_id", view.item_id),
                judge_name=self.name,
                score=(self.score_min + self.score_max) / 2,
                rationale=f"(anthropic-error fallback) {type(e).__name__}: {e}",
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
            cost_usd=cost,
            latency_s=time.perf_counter() - t0,
            meta={"model": self.model, "input_tokens": usage.input_tokens,
                  "output_tokens": usage.output_tokens},
        )

    def judge_batch(self, requests: list[RubricRequest]) -> list[JudgeOutput]:
        """Threaded batch — Anthropic SDK is thread-safe; threads parallelize HTTP wait."""
        if len(requests) == 1:
            return [self.judge_one(requests[0])]
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            return list(pool.map(self.judge_one, requests))


# Approx pricing as of 2026 — update via env override if needed.
_PRICING_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"in": 1.0, "out": 5.0, "cache_read": 0.10, "cache_write": 1.25},
    "claude-sonnet-4-5": {"in": 3.0, "out": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-opus-4-7": {"in": 15.0, "out": 75.0, "cache_read": 1.50, "cache_write": 18.75},
}


def _estimate_cost(model: str, in_tok: int, out_tok: int, cache_read: int, cache_write: int) -> float:
    base = None
    for k, v in _PRICING_PER_MTOK.items():
        if k in model:
            base = v
            break
    if base is None:
        return 0.0
    return (
        in_tok * base["in"] / 1_000_000
        + out_tok * base["out"] / 1_000_000
        + cache_read * base["cache_read"] / 1_000_000
        + cache_write * base["cache_write"] / 1_000_000
    )


def build_anthropic_ensemble(
    primary_model: str = "claude-haiku-4-5",
    diverse_model: str = "claude-sonnet-4-5",
    warm_temperature: float = 0.6,
) -> list[AnthropicJudge]:
    """Three-judge Anthropic ensemble for single-key real runs."""
    return [
        AnthropicJudge(name=f"{primary_model}@T0", model=primary_model, temperature_default=0.0),
        AnthropicJudge(name=f"{diverse_model}@T0", model=diverse_model, temperature_default=0.0),
        AnthropicJudge(name=f"{primary_model}@T{warm_temperature}",
                       model=primary_model, temperature_default=warm_temperature),
    ]
