"""Ordinal conformal prediction for LLM/VLM judge scores.

Implements the boundary-adjusted split-conformal procedure of
Sheng et al., *Analyzing Uncertainty of LLM-as-a-Judge* (EMNLP 2025 /
arXiv 2509.18658). Given a calibration set with golden labels and predicted
point estimates (e.g. from the Tweedie denoiser), we form a prediction interval
[ℓ, u] for each test item that *covers the true latent score with probability
≥ 1 − α*.

We deliberately implement this from scratch rather than wrap MAPIE so that:
  (a) the package is importable without MAPIE installed (it is an optional
      extra), and
  (b) the ordinal boundary adjustment described in §3.3 of the paper is
      transparent to readers / reviewers.

If MAPIE 0.8.6 is available, callers can use it via the `mapie` integration
in scripts/calibrate.py — both paths produce identical intervals up to
floating-point rounding when nonconformity = abs residual.

Posterior-variance-aware nonconformity
--------------------------------------
The key insight from the design's §4.3(c): the Tweedie denoiser supplies a
per-item posterior variance σ̂². We use it as a nonconformity *scaler*:

    α_i = |s̃_i − ŝ_i| / max(σ̂_i, ε)

This makes the interval *adaptive* — uncertain items get wider intervals,
confident ones tighter — at the cost of nothing extra during calibration.
This matches the standard "adaptive" or "studentized" split-conformal recipe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalInterval:
    item_id: str
    point_estimate: float
    lower: float
    upper: float
    width: float
    nonconformity: float


@dataclass
class OrdinalConformalResult:
    intervals: list[ConformalInterval]
    coverage: float
    mean_width: float
    quantile: float


class OrdinalBoundaryConformal:
    """Studentized split-conformal with ordinal-grid boundary snap."""

    def __init__(
        self,
        alpha: float = 0.10,
        score_classes: tuple[int, ...] = (1, 2, 3, 4, 5),
        adaptive: bool = True,
        boundary_snap: bool = True,
        eps: float = 1e-3,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = float(alpha)
        self.score_classes = tuple(score_classes)
        self.adaptive = bool(adaptive)
        self.boundary_snap = bool(boundary_snap)
        self.eps = float(eps)
        self._q: float | None = None

    def fit(
        self,
        cal_predictions: np.ndarray,
        cal_labels: np.ndarray,
        cal_sigmas: np.ndarray | None = None,
    ) -> "OrdinalBoundaryConformal":
        cal_predictions = np.asarray(cal_predictions, dtype=np.float64).ravel()
        cal_labels = np.asarray(cal_labels, dtype=np.float64).ravel()
        if cal_predictions.size != cal_labels.size:
            raise ValueError("cal_predictions and cal_labels length mismatch")
        if cal_predictions.size < 10:
            raise ValueError(
                f"Need ≥10 calibration items for stable q̂; got {cal_predictions.size}"
            )

        residuals = np.abs(cal_labels - cal_predictions)
        if self.adaptive:
            if cal_sigmas is None:
                raise ValueError("adaptive=True requires cal_sigmas")
            sigmas = np.asarray(cal_sigmas, dtype=np.float64).ravel()
            scaled = residuals / np.maximum(sigmas, self.eps)
        else:
            scaled = residuals

        # Finite-sample-corrected quantile per Vovk et al.
        n = scaled.size
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = float(np.clip(q_level, 0.0, 1.0))
        self._q = float(np.quantile(scaled, q_level, method="higher"))
        return self

    def predict(
        self,
        item_ids: list[str],
        predictions: np.ndarray,
        sigmas: np.ndarray | None = None,
    ) -> list[ConformalInterval]:
        if self._q is None:
            raise RuntimeError("Call .fit(...) before .predict(...)")
        predictions = np.asarray(predictions, dtype=np.float64).ravel()
        if self.adaptive:
            if sigmas is None:
                raise ValueError("adaptive=True requires sigmas at predict time")
            sigmas = np.asarray(sigmas, dtype=np.float64).ravel()
            half = self._q * np.maximum(sigmas, self.eps)
        else:
            half = np.full_like(predictions, self._q)

        lo = predictions - half
        hi = predictions + half

        if self.boundary_snap:
            lo = self._snap_lower(lo)
            hi = self._snap_upper(hi)

        return [
            ConformalInterval(
                item_id=iid,
                point_estimate=float(predictions[i]),
                lower=float(lo[i]),
                upper=float(hi[i]),
                width=float(hi[i] - lo[i]),
                nonconformity=float(half[i]),
            )
            for i, iid in enumerate(item_ids)
        ]

    def evaluate(
        self,
        item_ids: list[str],
        predictions: np.ndarray,
        labels: np.ndarray,
        sigmas: np.ndarray | None = None,
    ) -> OrdinalConformalResult:
        intervals = self.predict(item_ids, predictions, sigmas=sigmas)
        labels = np.asarray(labels, dtype=np.float64).ravel()
        covered = np.array([
            iv.lower <= labels[i] <= iv.upper for i, iv in enumerate(intervals)
        ])
        return OrdinalConformalResult(
            intervals=intervals,
            coverage=float(covered.mean()),
            mean_width=float(np.mean([iv.width for iv in intervals])),
            quantile=float(self._q or 0.0),
        )

    def _snap_lower(self, x: np.ndarray) -> np.ndarray:
        cls = np.asarray(self.score_classes, dtype=np.float64)
        floor_class = cls.min() - 0.5
        out = np.maximum(x, floor_class)
        # Snap upward to the nearest class-boundary so the interval lower edge
        # corresponds to a meaningful ordinal threshold (Sheng et al. §3.3).
        snapped = np.empty_like(out)
        for i, v in enumerate(out):
            below = cls[cls <= v + 0.5]
            snapped[i] = (below.max() - 0.5) if below.size else floor_class
        return snapped

    def _snap_upper(self, x: np.ndarray) -> np.ndarray:
        cls = np.asarray(self.score_classes, dtype=np.float64)
        ceil_class = cls.max() + 0.5
        out = np.minimum(x, ceil_class)
        snapped = np.empty_like(out)
        for i, v in enumerate(out):
            above = cls[cls >= v - 0.5]
            snapped[i] = (above.min() + 0.5) if above.size else ceil_class
        return snapped
