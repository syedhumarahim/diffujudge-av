"""Base judge ABC and structured input/output schemas.

A judge takes a (PromptView, scoring scale) request and returns a structured
output: a scalar score, optional per-criterion scores, a rationale, and
optionally log-probabilities of the score-token. Backends honor
PromptView.temperature and PromptView.score_id_format so the perturbation
cascade can drive them without backend-specific glue.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from diffujudge.perturbations.operators import PromptView


@dataclass
class RubricRequest:
    """A single judgment request: which view, which scale, optional reference."""

    view: PromptView
    score_scale: int = 5
    reference_answer: str | None = None
    candidate_answer: str | None = None
    pairwise_other: str | None = None  # for pairwise comparison judges


@dataclass
class JudgeOutput:
    """Structured judgment from one judge on one request."""

    item_id: str
    sample_id: str
    judge_name: str
    score: float                         # primary scalar in [score_min, score_max]
    rubric_scores: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    score_logprobs: dict[str, float] | None = None
    raw_response: str = ""
    cost_usd: float = 0.0
    latency_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class BaseJudge(abc.ABC):
    """Subclass this for new backends. Implement `judge_one`; batching default-loops."""

    name: str

    def __init__(self, name: str, score_min: float = 1.0, score_max: float = 5.0) -> None:
        self.name = name
        self.score_min = float(score_min)
        self.score_max = float(score_max)

    @abc.abstractmethod
    def judge_one(self, request: RubricRequest) -> JudgeOutput:
        """Return one structured judgment."""

    def judge_batch(self, requests: list[RubricRequest]) -> list[JudgeOutput]:
        """Default: serial loop. Override for true batched backends (vLLM, anthropic batch)."""
        return [self.judge_one(r) for r in requests]

    def clip(self, score: float) -> float:
        return max(self.score_min, min(self.score_max, float(score)))
