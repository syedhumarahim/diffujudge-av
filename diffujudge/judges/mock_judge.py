"""Mock judge with controllable, *bias-injecting* behavior — used for tests
and for end-to-end pipeline runs without GPU/API access.

The mock simulates each of the documented bias sources at known magnitude so
the eval-of-eval harness has something nontrivial to recover. Determined by
seed → reproducible, but the noise distribution per perturbation level is
configured to roughly match what the literature reports.
"""
from __future__ import annotations

import hashlib
import random
import time
from typing import Any

from diffujudge.judges.base import BaseJudge, JudgeOutput, RubricRequest


def _hash_to_unit(*parts: Any) -> float:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class MockJudge(BaseJudge):
    """Returns gold + per-bias-source noise.

    `gold_lookup` maps item_id → ground-truth latent score. If absent, a
    deterministic gold is derived from the item_id hash.
    """

    BIAS_SIGMA = {
        # sigma in score units that each level injects
        0: 0.20,   # anchor
        1: 0.30,   # option swap (position)
        2: 0.25,   # rubric paraphrase
        3: 0.20,   # criterion reorder
        4: 0.35,   # score-ID swap
        5: 0.40,   # temperature
        6: 0.20,   # exemplar resample
        7: 0.30,   # frame shuffle
    }
    BIAS_BIAS = {
        # systematic offset per level (some biases are not zero-mean)
        4: -0.15,  # Arabic→Roman tends to compress toward middle
        5: 0.0,
    }

    def __init__(
        self,
        name: str = "mock-judge",
        score_min: float = 1.0,
        score_max: float = 5.0,
        gold_lookup: dict[str, float] | None = None,
        family_bias: float = 0.0,
        verbosity_slope: float = 0.0,
    ) -> None:
        super().__init__(name=name, score_min=score_min, score_max=score_max)
        self.gold_lookup = gold_lookup or {}
        self.family_bias = float(family_bias)        # self-preference bias
        self.verbosity_slope = float(verbosity_slope)

    def _gold_for(self, item_id: str) -> float:
        if item_id in self.gold_lookup:
            return float(self.gold_lookup[item_id])
        u = _hash_to_unit(item_id, "gold")
        return self.score_min + u * (self.score_max - self.score_min)

    def judge_one(self, request: RubricRequest) -> JudgeOutput:
        t0 = time.perf_counter()
        view = request.view
        gold = self._gold_for(view.item_id)

        level = int(view.meta.get("perturb_level", 0))
        for lvl in range(8):
            if view.meta.get(f"perturb_{lvl}", False):
                level = lvl
                break

        sigma = self.BIAS_SIGMA.get(level, 0.2)
        sys_bias = self.BIAS_BIAS.get(level, 0.0)

        # Length signal (verbosity bias) — fall back to rubric length as a proxy.
        length = sum(len(c) for c in view.rubric)
        verbosity_term = self.verbosity_slope * (length - 200) / 200.0

        rng = random.Random(int(_hash_to_unit(view.item_id, level, view.score_id_format, view.temperature) * 2**31))
        noise = rng.gauss(0.0, sigma) * (1.0 + 0.5 * view.temperature)

        score = gold + sys_bias + verbosity_term + self.family_bias + noise
        score = self.clip(score)

        return JudgeOutput(
            item_id=view.item_id,
            sample_id=view.meta.get("sample_id", view.item_id),
            judge_name=self.name,
            score=score,
            rubric_scores={f"crit_{i}": score for i in range(len(view.rubric))},
            rationale="(mock) deterministic gold + per-level Gaussian noise",
            raw_response=f"score={score:.3f}",
            cost_usd=0.0,
            latency_s=time.perf_counter() - t0,
            meta={"gold": gold, "level": level, "sigma": sigma},
        )
