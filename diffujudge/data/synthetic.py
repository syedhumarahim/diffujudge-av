"""Synthetic AV-flavored dataset for offline development and CI.

Generates `n` items with:
  - a behavior label drawn from BEHAVIOR_CATEGORIES_SUBSET,
  - a question / reference / candidate answer triplet templated on the label,
  - a deterministic latent gold score (via hash of item_id),
  - ASCII-frame stand-ins so the rest of the pipeline does not branch on
    "do we have video?".

This dataset's purpose is to exercise every code path end-to-end without
external data. All numbers are reproducible from `seed`.
"""
from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field

from diffujudge.taxonomy.nhtsa import BEHAVIOR_CATEGORIES_SUBSET, classify_label


@dataclass
class SyntheticItem:
    item_id: str
    question: str
    reference_answer: str
    candidate_answer: str
    behavior_label: str
    gold_score: float
    frames: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


_QUESTION_TEMPLATES = {
    "cut_in": "Did the silver sedan cut into the ego vehicle's lane within 2 seconds time-to-collision?",
    "hard_brake": "Is the lead vehicle decelerating at more than 0.4g?",
    "vru_conflict": "Is there a pedestrian or cyclist in or near the ego vehicle's path?",
    "left_turn_oncoming": "Is the ego vehicle making an unprotected left turn with oncoming traffic?",
    "tailgating": "Is the ego vehicle following the lead vehicle at THW < 1.0 second?",
    "no_conflict": "Describe the ego vehicle's near-term plan in this scene.",
}

_REFERENCE_ANSWERS = {
    "cut_in": "Yes — the silver sedan from the right lane crosses the lane line at ~1.5s TTC; ego must brake.",
    "hard_brake": "Yes — brake lights and rapid range closure indicate >0.4g deceleration.",
    "vru_conflict": "Yes — a pedestrian is mid-crosswalk in the ego's intended path.",
    "left_turn_oncoming": "Yes — ego is in the protected-left bay turning across two oncoming lanes.",
    "tailgating": "Yes — measured headway is approximately 0.7 seconds, below the 1.0s threshold.",
    "no_conflict": "Continue at current speed; lane-keep with the right lane.",
}


def _candidate_for(label: str, gold: float, rng: random.Random) -> str:
    """Return a candidate whose verbosity correlates loosely with quality."""
    base = _REFERENCE_ANSWERS[label]
    if gold >= 4.0:
        return base + " " + rng.choice(["Confidence is high.", "Multiple cues agree."])
    if gold >= 3.0:
        return base.split(";")[0] + "."
    return rng.choice(["Maybe.", "Unclear.", "Cannot tell from this frame."])


def generate_synthetic_corpus(
    n: int = 200,
    seed: int = 42,
    n_frames: int = 6,
) -> list[SyntheticItem]:
    rng = random.Random(seed)
    out: list[SyntheticItem] = []
    for i in range(n):
        label = rng.choice(BEHAVIOR_CATEGORIES_SUBSET)
        item_id = f"syn_{i:04d}_{label}"
        # Latent gold drawn near the safety-critical mid-high range,
        # with a few low-quality candidates as hard negatives.
        gold = rng.choices([5.0, 4.0, 3.0, 2.0, 1.0], weights=[3, 4, 4, 2, 1])[0]
        gold += rng.uniform(-0.3, 0.3)
        gold = max(1.0, min(5.0, gold))
        frames = [f"synthetic://{item_id}/frame_{f:02d}" for f in range(n_frames)]
        out.append(
            SyntheticItem(
                item_id=item_id,
                question=_QUESTION_TEMPLATES[label],
                reference_answer=_REFERENCE_ANSWERS[label],
                candidate_answer=_candidate_for(label, gold, rng),
                behavior_label=label,
                gold_score=gold,
                frames=frames,
                meta={
                    "is_safety_critical": classify_label(label).is_safety_critical,
                    "synthetic": True,
                },
            )
        )
    return out


@dataclass
class SyntheticDataset:
    items: list[SyntheticItem]

    @classmethod
    def build(cls, n: int = 200, seed: int = 42, n_frames: int = 6) -> "SyntheticDataset":
        return cls(items=generate_synthetic_corpus(n=n, seed=seed, n_frames=n_frames))

    def __iter__(self) -> Iterator[SyntheticItem]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def gold_lookup(self) -> dict[str, float]:
        return {it.item_id: it.gold_score for it in self.items}
