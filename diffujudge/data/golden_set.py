"""Three-tier golden set construction.

Per the design's §6:

  Tier 1 — public Lingo-Judge anchors (free): treat any LingoQA item with
           a Lingo-Judge confidence ≥ 0.8 as gold.
  Tier 2 — synthetic supermajority (cheap): items where ≥3 closed judges
           (GPT-4o, Claude 3.5/4, Gemini 2.5) agree at κ > 0.8.
  Tier 3 — manual mini-set (high signal): 30–50 hand-labeled corner cases.

This module assembles the three tiers into a single `GoldenSet` with a
`provenance` field per item so downstream analyses can stratify by tier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from diffujudge.metrics.agreement import cohen_kappa


Provenance = Literal["lingo_judge", "supermajority", "manual"]


@dataclass
class GoldenItem:
    item_id: str
    score: float
    provenance: Provenance
    confidence: float = 1.0
    notes: str = ""


@dataclass
class GoldenSet:
    items: list[GoldenItem]
    calibration_ids: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)

    def lookup(self) -> dict[str, GoldenItem]:
        return {g.item_id: g for g in self.items}

    def split(self, calibration_frac: float = 0.6, seed: int = 42) -> "GoldenSet":
        rng = np.random.default_rng(seed)
        ids = [g.item_id for g in self.items]
        rng.shuffle(ids)
        n_cal = int(round(len(ids) * calibration_frac))
        return GoldenSet(
            items=self.items,
            calibration_ids=ids[:n_cal],
            test_ids=ids[n_cal:],
        )


def build_three_tier_gold(
    tier1_lingo: dict[str, tuple[float, float]] | None = None,
    tier2_judge_scores: dict[str, dict[str, float]] | None = None,
    tier3_manual: dict[str, float] | None = None,
    *,
    tier1_confidence_threshold: float = 0.8,
    tier2_supermajority_kappa: float = 0.8,
) -> GoldenSet:
    """Compose the three tiers into one GoldenSet.

    Args:
        tier1_lingo: item_id → (score, lingo_judge_confidence)
        tier2_judge_scores: item_id → judge_name → score (need ≥3 judges)
        tier3_manual: item_id → human-assigned score
    """
    items: list[GoldenItem] = []

    if tier1_lingo:
        for iid, (s, conf) in tier1_lingo.items():
            if conf >= tier1_confidence_threshold:
                items.append(
                    GoldenItem(item_id=iid, score=float(s), provenance="lingo_judge", confidence=float(conf))
                )

    if tier2_judge_scores:
        # Supermajority: pairwise κ across judges ≥ threshold.
        for iid, judge_to_score in tier2_judge_scores.items():
            if iid in {g.item_id for g in items}:
                continue
            judges = list(judge_to_score)
            if len(judges) < 3:
                continue
            scores = np.array([judge_to_score[j] for j in judges])
            # Quantize to integer grid for κ; keep continuous mean as score.
            quantized = np.round(scores).astype(int)
            kappas: list[float] = []
            for i in range(len(judges)):
                for j in range(i + 1, len(judges)):
                    try:
                        kappas.append(cohen_kappa(quantized[i:i+1], quantized[j:j+1], weights=None))
                    except (ValueError, IndexError):
                        pass
            if kappas and float(np.mean(kappas)) >= tier2_supermajority_kappa:
                items.append(
                    GoldenItem(
                        item_id=iid,
                        score=float(scores.mean()),
                        provenance="supermajority",
                        confidence=float(np.mean(kappas)) if kappas else 1.0,
                    )
                )

    if tier3_manual:
        existing = {g.item_id: g for g in items}
        for iid, s in tier3_manual.items():
            existing[iid] = GoldenItem(item_id=iid, score=float(s), provenance="manual", confidence=1.0)
        items = list(existing.values())

    return GoldenSet(items=items)
