"""Calibration metrics: ECE, MCE, Brier.

These work for ordinal-score judges by binning the *predicted score* and
comparing it to the gold score within each bin. For binary correctness
labels, pass a probability and a 0/1 label. For ordinal scoring, the
reliability diagram is per-bin (predicted_mean vs. gold_mean).
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    score_min: float = 1.0,
    score_max: float = 5.0,
) -> float:
    """Ordinal-friendly ECE.

    Bin items by the predicted score; in each bin, ECE-contribution is
    |bin_pred_mean − bin_label_mean|, weighted by bin frequency. The result is
    on the same units as the score, then normalized by (score_max − score_min)
    to land in [0, 1].
    """
    p = np.asarray(predictions, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if p.size == 0:
        return float("nan")
    edges = np.linspace(score_min, score_max, n_bins + 1)
    edges[-1] += 1e-9
    idx = np.digitize(p, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    total = 0.0
    n = p.size
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        bin_pred = p[mask].mean()
        bin_lab = y[mask].mean()
        total += (mask.sum() / n) * abs(bin_pred - bin_lab)
    return float(total / (score_max - score_min))


def mce(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    score_min: float = 1.0,
    score_max: float = 5.0,
) -> float:
    """Maximum calibration error — the *worst* bin gap, normalized to [0, 1]."""
    p = np.asarray(predictions, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if p.size == 0:
        return float("nan")
    edges = np.linspace(score_min, score_max, n_bins + 1)
    edges[-1] += 1e-9
    idx = np.digitize(p, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    gaps: list[float] = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        gaps.append(abs(p[mask].mean() - y[mask].mean()))
    if not gaps:
        return float("nan")
    return float(max(gaps) / (score_max - score_min))


def brier_score(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Continuous Brier (mean-squared-error in score units)."""
    p = np.asarray(predictions, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    return float(np.mean((p - y) ** 2))


def reliability_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    score_min: float = 1.0,
    score_max: float = 5.0,
) -> dict[str, np.ndarray]:
    """Bin centers + per-bin (mean_pred, mean_label, count) for the reliability diagram."""
    p = np.asarray(predictions, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    edges = np.linspace(score_min, score_max, n_bins + 1)
    edges[-1] += 1e-9
    idx = np.digitize(p, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pred_means = np.zeros(n_bins)
    lab_means = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            pred_means[b] = p[mask].mean()
            lab_means[b] = y[mask].mean()
            counts[b] = int(mask.sum())
    return {
        "centers": centers,
        "pred_means": pred_means,
        "label_means": lab_means,
        "counts": counts,
    }
