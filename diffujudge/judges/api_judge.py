"""litellm-backed API judge for closed-source baselines (GPT-4o, Claude, Gemini).

These are the design's "API fallback for closed-judge baseline." They are
useful for:
  (a) running the pipeline end-to-end without a GPU,
  (b) the eval-of-eval comparison ("our 7B Tweedie ensemble matches GPT-4o
      on κ"),
  (c) Tier-2 supermajority synthetic-gold construction.

Set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` in `.env` to
enable each provider; missing keys raise at instantiation, not at runtime.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from diffujudge.judges.base import BaseJudge, JudgeOutput, RubricRequest
from diffujudge.perturbations.operators import render_score_ids


_SCORE_PATTERNS = [
    re.compile(r'"score"\s*:\s*([0-9.]+)'),
    re.compile(r"score[:\s=]+([0-9.]+)"),
    re.compile(r"\b([1-9](?:\.[0-9]+)?)\s*(?:/\s*5|out of 5)\b", re.IGNORECASE),
]
_ROMAN_TO_INT = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
_ALPHA_TO_INT = {c: i + 1 for i, c in enumerate("ABCDE")}


def parse_score_from_text(text: str, fmt: str = "arabic", default: float = 3.0) -> float:
    """Best-effort score extractor robust to JSON, plaintext, and ID-format variants."""
    if not text:
        return default
    if fmt == "roman":
        for tok, val in _ROMAN_TO_INT.items():
            if re.search(rf"\b{tok}\b", text):
                return float(val)
    if fmt == "alpha":
        m = re.search(r"\b([A-E])\b", text)
        if m:
            return float(_ALPHA_TO_INT.get(m.group(1), default))
    for pat in _SCORE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    try:
        obj = json.loads(text[text.find("{") : text.rfind("}") + 1])
        if isinstance(obj, dict) and "score" in obj:
            return float(obj["score"])
    except (ValueError, json.JSONDecodeError):
        pass
    return default


def _build_prompt(req: RubricRequest) -> str:
    view = req.view
    ids = render_score_ids(req.score_scale, view.score_id_format)
    rubric_block = "\n".join(f"- {c}" for c in view.rubric)
    exemplar_block = ""
    if view.exemplars:
        exemplar_block = "\n\nExamples:\n" + "\n".join(
            f"Q: {e.get('question','')}\nA: {e.get('answer','')}\nScore: {e.get('score','')}"
            for e in view.exemplars
        )
    candidate = req.candidate_answer or "(no candidate provided)"
    reference = f"\nReference answer: {req.reference_answer}" if req.reference_answer else ""
    return (
        "You are evaluating a model's answer to an autonomous-driving question.\n"
        f"Use the following ordinal scale: {ids[0]} (worst) to {ids[-1]} (best).\n"
        f"Question: {view.question}\n"
        f"Rubric:\n{rubric_block}{reference}{exemplar_block}\n\n"
        f"Candidate answer: {candidate}\n\n"
        'Return STRICT JSON: {"score": <number>, "rationale": "<one sentence>"}'
    )


class LiteLLMJudge(BaseJudge):
    """Generic API judge via litellm. Provider auto-routed by model string."""

    def __init__(
        self,
        name: str,
        model: str,
        score_min: float = 1.0,
        score_max: float = 5.0,
        temperature_default: float = 0.0,
        max_tokens: int = 256,
        api_base: str | None = None,
    ) -> None:
        super().__init__(name=name, score_min=score_min, score_max=score_max)
        self.model = model
        self.temperature_default = temperature_default
        self.max_tokens = max_tokens
        self.api_base = api_base
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "LiteLLMJudge requires `pip install diffujudge-av[api]`"
            ) from e
        self._check_keys()

    def _check_keys(self) -> None:
        m = self.model.lower()
        if "together" in m:
            if not os.getenv("TOGETHER_API_KEY"):
                raise OSError("TOGETHER_API_KEY not set")
            return
        if m.startswith("gpt") or "openai" in m:
            if not os.getenv("OPENAI_API_KEY"):
                raise OSError("OPENAI_API_KEY not set")
        if "claude" in m or "anthropic" in m:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise OSError("ANTHROPIC_API_KEY not set")
        if "gemini" in m or "google" in m:
            if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
                raise OSError("GOOGLE_API_KEY (or GEMINI_API_KEY) not set")

    def judge_one(self, request: RubricRequest) -> JudgeOutput:
        import litellm

        t0 = time.perf_counter()
        prompt = _build_prompt(request)
        view = request.view
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=view.temperature or self.temperature_default,
            max_tokens=self.max_tokens,
        )
        if self.api_base:
            kwargs["api_base"] = self.api_base
        try:
            resp = litellm.completion(**kwargs)
            text = resp.choices[0].message.content or ""
            cost = float(getattr(resp, "_response_cost", 0.0) or 0.0)
        except Exception as e:  # noqa: BLE001 — surface but do not crash the whole run
            return JudgeOutput(
                item_id=view.item_id,
                sample_id=view.meta.get("sample_id", view.item_id),
                judge_name=self.name,
                score=(self.score_min + self.score_max) / 2,
                rationale=f"(api-error fallback) {type(e).__name__}: {e}",
                raw_response="",
                latency_s=time.perf_counter() - t0,
                meta={"error": True},
            )

        score = parse_score_from_text(text, fmt=view.score_id_format)
        score = self.clip(score)
        return JudgeOutput(
            item_id=view.item_id,
            sample_id=view.meta.get("sample_id", view.item_id),
            judge_name=self.name,
            score=score,
            rationale=text[:500],
            raw_response=text,
            cost_usd=cost,
            latency_s=time.perf_counter() - t0,
            meta={"model": self.model},
        )
