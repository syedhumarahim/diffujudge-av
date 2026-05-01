"""Multi-judge ensemble: route a batch of requests across N judges and
aggregate. Default aggregation is simple mean — the cascade + Tweedie path is
the *real* aggregation; this is the lightweight baseline for the bias-delta
tables.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from diffujudge.judges.base import BaseJudge, JudgeOutput, RubricRequest


@dataclass
class EnsembleResult:
    item_id: str
    sample_id: str
    mean_score: float
    median_score: float
    std_score: float
    per_judge: dict[str, float] = field(default_factory=dict)


class JudgeEnsemble:
    def __init__(self, judges: list[BaseJudge]) -> None:
        if not judges:
            raise ValueError("Need at least one judge")
        self.judges = judges

    @property
    def names(self) -> list[str]:
        return [j.name for j in self.judges]

    def judge(self, requests: list[RubricRequest]) -> tuple[list[JudgeOutput], list[EnsembleResult]]:
        flat: list[JudgeOutput] = []
        for j in self.judges:
            flat.extend(j.judge_batch(requests))

        bucket: dict[str, list[JudgeOutput]] = defaultdict(list)
        for o in flat:
            bucket[o.sample_id].append(o)

        agg: list[EnsembleResult] = []
        for sample_id, outs in bucket.items():
            scores = np.array([o.score for o in outs], dtype=np.float64)
            agg.append(
                EnsembleResult(
                    item_id=outs[0].item_id,
                    sample_id=sample_id,
                    mean_score=float(scores.mean()),
                    median_score=float(np.median(scores)),
                    std_score=float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
                    per_judge={o.judge_name: float(o.score) for o in outs},
                )
            )
        return flat, agg
