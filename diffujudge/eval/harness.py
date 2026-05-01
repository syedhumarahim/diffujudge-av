"""Eval-of-eval harness — measures the *judge*, not the model under test.

This is the marketable contribution per the design's §2.6: position bias,
verbosity bias, scoring-ID bias, stochastic stability, plus the standard
agreement and calibration suite. Reports are JSON-serializable so they slot
straight into wandb / a dashboard / a markdown table for the README.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from diffujudge.metrics.agreement import (
    cohen_kappa,
    kendall_tau,
    krippendorff_alpha,
    pearson,
    spearman,
)
from diffujudge.metrics.bias import (
    position_bias_delta,
    scoring_id_bias_delta,
    stochastic_stability,
)
from diffujudge.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    mce,
    reliability_curve,
)


@dataclass
class EvalReport:
    n_items: int
    cohen_kappa: float
    krippendorff_alpha: float
    pearson: float
    spearman: float
    kendall_tau: float
    ece_baseline: float
    ece_denoised: float
    mce_denoised: float
    brier_baseline: float
    brier_denoised: float
    position_bias_delta: float | None = None
    scoring_id_bias_delta: float | None = None
    stochastic_stability: float | None = None
    conformal_coverage: float | None = None
    interval_mean_width: float | None = None
    reliability: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=lambda o: list(o) if hasattr(o, "tolist") else str(o))


class EvalOfEvalHarness:
    def __init__(self, score_classes: tuple[int, ...] = (1, 2, 3, 4, 5)) -> None:
        self.score_classes = tuple(score_classes)

    def run(
        self,
        item_ids: list[str],
        denoised_predictions: np.ndarray,
        raw_predictions: np.ndarray,
        gold: dict[str, float],
        *,
        position_pairs: tuple[np.ndarray, np.ndarray] | None = None,
        scoring_id_pairs: tuple[np.ndarray, np.ndarray] | None = None,
        seeds_matrix: np.ndarray | None = None,
        conformal_coverage: float | None = None,
        interval_mean_width: float | None = None,
    ) -> EvalReport:
        gold_arr = np.array([gold[i] for i in item_ids if i in gold], dtype=np.float64)
        denoised_arr = np.array(
            [p for i, p in zip(item_ids, denoised_predictions) if i in gold],
            dtype=np.float64,
        )
        raw_arr = np.array(
            [p for i, p in zip(item_ids, raw_predictions) if i in gold],
            dtype=np.float64,
        )

        if gold_arr.size == 0:
            raise ValueError("No item_ids overlap with gold")

        score_min = float(min(self.score_classes)) - 0.5
        score_max = float(max(self.score_classes)) + 0.5

        # Quantize for κ.
        gold_q = np.clip(np.round(gold_arr).astype(int), self.score_classes[0], self.score_classes[-1])
        denoised_q = np.clip(np.round(denoised_arr).astype(int), self.score_classes[0], self.score_classes[-1])

        # Krippendorff-α expects (n_raters, n_items). Two raters: gold, denoised.
        kr_data = np.stack([gold_q.astype(float), denoised_q.astype(float)], axis=0)
        try:
            kr_alpha = krippendorff_alpha(kr_data, level="ordinal")
        except (ValueError, ZeroDivisionError):
            kr_alpha = float("nan")

        return EvalReport(
            n_items=int(gold_arr.size),
            cohen_kappa=cohen_kappa(gold_q, denoised_q, weights="quadratic"),
            krippendorff_alpha=float(kr_alpha),
            pearson=pearson(gold_arr, denoised_arr),
            spearman=spearman(gold_arr, denoised_arr),
            kendall_tau=kendall_tau(gold_arr, denoised_arr),
            ece_baseline=expected_calibration_error(raw_arr, gold_arr, score_min=score_min, score_max=score_max),
            ece_denoised=expected_calibration_error(denoised_arr, gold_arr, score_min=score_min, score_max=score_max),
            mce_denoised=mce(denoised_arr, gold_arr, score_min=score_min, score_max=score_max),
            brier_baseline=brier_score(raw_arr, gold_arr),
            brier_denoised=brier_score(denoised_arr, gold_arr),
            position_bias_delta=(
                position_bias_delta(*position_pairs) if position_pairs else None
            ),
            scoring_id_bias_delta=(
                scoring_id_bias_delta(*scoring_id_pairs) if scoring_id_pairs else None
            ),
            stochastic_stability=(
                stochastic_stability(seeds_matrix) if seeds_matrix is not None else None
            ),
            conformal_coverage=conformal_coverage,
            interval_mean_width=interval_mean_width,
            reliability={
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in reliability_curve(
                    denoised_arr, gold_arr, score_min=score_min, score_max=score_max
                ).items()
            },
        )
