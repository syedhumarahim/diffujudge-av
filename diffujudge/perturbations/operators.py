"""Forward-diffusion perturbation operators.

Each operator corresponds to one canonical bias source from the 2024–25
LLM-as-a-Judge literature, operationalized as a noise injection step:

    t=1   option_swap          — Shi et al., IJCNLP-AACL 2025 (position bias)
    t=2   rubric_paraphrase    — Gao et al., SPUQ (arXiv 2403.02509)
    t=3   criterion_reorder    — Chen et al. (arXiv 2506.22316), rubric order
    t=4   score_id_swap        — Chen et al., score ID format (Arabic↔Roman)
    t=5   temperature_noise    — Thakur et al., Rating Roulette (arXiv 2510.27106)
    t=6   exemplar_resample    — few-shot variance
    t=7   frame_shuffle        — video-only; tests temporal robustness

The operators are *purely functional* on a `PromptView` — they do not call any
model. Sampling N×k score observations is the cascade's job (cascade.py).
"""
from __future__ import annotations

import random
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

PerturbationOperator = Callable[["PromptView", random.Random], "PromptView"]


@dataclass
class PromptView:
    """A judge-ready view of an item: question, frames, options, rubric, formatting.

    The view is mutated by perturbation operators to produce N×k judge inputs.
    Frames are paths or PIL handles — operators never decode pixels.
    """

    item_id: str
    question: str
    rubric: list[str]                                  # criterion strings, in display order
    score_id_format: str = "arabic"                    # "arabic" | "roman" | "alpha"
    options: list[str] = field(default_factory=list)   # for pairwise / MCQ judges
    frames: list[str] = field(default_factory=list)    # ordered frame paths
    exemplars: list[dict[str, Any]] = field(default_factory=list)  # few-shot pool
    n_exemplars: int = 2                               # how many to render
    temperature: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "PromptView":
        return deepcopy(self)


# ----- t=1 -----
def option_swap(view: PromptView, rng: random.Random) -> PromptView:
    """Reverse option ordering. Mitigation by Wang et al. 2024c (order-swap averaging)."""
    if not view.options:
        return view
    out = view.copy()
    out.options = list(reversed(out.options))
    out.meta["perturb_option_swap"] = True
    return out


# ----- t=2 -----
_PARAPHRASE_BANK: dict[str, list[str]] = {}


def register_paraphrases(rubric_id: str, variants: list[str]) -> None:
    """Pre-cache LLM-generated paraphrases. The design specifies offline generation."""
    _PARAPHRASE_BANK[rubric_id] = list(variants)


def rubric_paraphrase(view: PromptView, rng: random.Random) -> PromptView:
    """Replace each rubric criterion with a cached paraphrase variant."""
    out = view.copy()
    new_rubric: list[str] = []
    for i, criterion in enumerate(out.rubric):
        key = out.meta.get("rubric_id", view.item_id) + f":crit{i}"
        bank = _PARAPHRASE_BANK.get(key)
        if bank:
            new_rubric.append(rng.choice(bank))
        else:
            new_rubric.append(_inline_paraphrase(criterion, rng))
    out.rubric = new_rubric
    out.meta["perturb_rubric_paraphrase"] = True
    return out


def _inline_paraphrase(text: str, rng: random.Random) -> str:
    """Cheap deterministic-ish hedging fallback when no LLM-cached bank is available."""
    hedges = ["Specifically: ", "In other words, ", "That is, ", "Equivalently, "]
    return rng.choice(hedges) + text[0].lower() + text[1:]


# ----- t=3 -----
def criterion_reorder(view: PromptView, rng: random.Random) -> PromptView:
    """Shuffle the order of rubric criteria. Chen et al. show this materially shifts scores."""
    if len(view.rubric) <= 1:
        return view
    out = view.copy()
    idx = list(range(len(out.rubric)))
    rng.shuffle(idx)
    out.rubric = [out.rubric[i] for i in idx]
    out.meta["perturb_criterion_reorder"] = idx
    return out


# ----- t=4 -----
_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
_ALPHA = list("ABCDEFGHIJ")


def score_id_swap(view: PromptView, rng: random.Random, target: str | None = None) -> PromptView:
    """Swap the score-ID rendering. Chen 2506.22316 documents materially different outputs
    between Arabic, Roman, and alphabetic IDs at otherwise-equal prompts."""
    out = view.copy()
    choices = ["arabic", "roman", "alpha"]
    if target is None:
        choices = [c for c in choices if c != out.score_id_format]
        target = rng.choice(choices)
    out.score_id_format = target
    out.meta["perturb_score_id_swap"] = target
    return out


def render_score_ids(n_classes: int, fmt: str) -> list[str]:
    if fmt == "arabic":
        return [str(i + 1) for i in range(n_classes)]
    if fmt == "roman":
        return _ROMAN[:n_classes]
    if fmt == "alpha":
        return _ALPHA[:n_classes]
    raise ValueError(f"Unknown score-id format: {fmt}")


# ----- t=5 -----
def temperature_noise(view: PromptView, rng: random.Random, grid: list[float] | None = None) -> PromptView:
    """Perturb sampling temperature only. The judge backend honors view.temperature."""
    out = view.copy()
    grid = grid or [0.0, 0.3, 0.7]
    out.temperature = rng.choice(grid)
    out.meta["perturb_temperature"] = out.temperature
    return out


# ----- t=6 -----
def exemplar_resample(view: PromptView, rng: random.Random) -> PromptView:
    """Resample which few-shot exemplars are rendered, holding the pool fixed."""
    if not view.exemplars:
        return view
    out = view.copy()
    k = min(out.n_exemplars, len(out.exemplars))
    out.exemplars = rng.sample(out.exemplars, k=k)
    out.meta["perturb_exemplar_resample"] = [e.get("id", str(i)) for i, e in enumerate(out.exemplars)]
    return out


# ----- t=7 -----
def frame_shuffle(view: PromptView, rng: random.Random) -> PromptView:
    """Video-only: shuffle frame order. Robust judges should be invariant to scrambling
    when the rubric is event-presence rather than temporal-ordering."""
    if len(view.frames) <= 1:
        return view
    out = view.copy()
    idx = list(range(len(out.frames)))
    rng.shuffle(idx)
    out.frames = [out.frames[i] for i in idx]
    out.meta["perturb_frame_shuffle"] = idx
    return out


OPERATOR_BY_LEVEL: dict[int, PerturbationOperator] = {
    1: option_swap,
    2: rubric_paraphrase,
    3: criterion_reorder,
    4: score_id_swap,
    5: temperature_noise,
    6: exemplar_resample,
    7: frame_shuffle,
}
