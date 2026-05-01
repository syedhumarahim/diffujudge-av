"""Bias-source robustness metrics.

These quantify *how much a judge's score moves* when one bias-source axis is
perturbed while everything else is held fixed. Each metric returns a single
scalar in score units (or σ-units for the stability metric), so they can be
compared across judges and across pre/post-denoising.

References:
- Position bias: Shi et al., IJCNLP-AACL 2025.
- Verbosity bias: Zheng et al. 2023 (MT-Bench), arXiv 2509.26072 ("Silent Judge").
- Scoring-ID bias: Chen et al., arXiv 2506.22316.
- Stochastic stability: Thakur et al., "Rating Roulette" (arXiv 2510.27106).
"""
from __future__ import annotations

import numpy as np


def position_bias_delta(score_a_first: np.ndarray, score_b_first: np.ndarray) -> float:
    """Mean |score(A,B) − score(B,A)| under option-order swap.

    Lower is better. Pass paired arrays of the *same* item under different
    option orderings.
    """
    a = np.asarray(score_a_first, dtype=np.float64).ravel()
    b = np.asarray(score_b_first, dtype=np.float64).ravel()
    return float(np.mean(np.abs(a - b)))


def verbosity_bias_delta(
    scores: np.ndarray,
    answer_lengths: np.ndarray,
) -> float:
    """Spearman correlation between answer length and score, holding quality
    constant. Implemented as a *partial-correlation-free proxy*: report the
    raw rank correlation and let the caller subtract out the gold-vs-length
    correlation if they have gold labels.
    """
    from scipy import stats

    s = np.asarray(scores, dtype=np.float64).ravel()
    L = np.asarray(answer_lengths, dtype=np.float64).ravel()
    if s.size < 3:
        return float("nan")
    r, _ = stats.spearmanr(s, L)
    return float(r)


def scoring_id_bias_delta(
    score_arabic: np.ndarray,
    score_roman: np.ndarray,
) -> float:
    """Mean |Arabic − Roman| score under otherwise-identical prompts."""
    a = np.asarray(score_arabic, dtype=np.float64).ravel()
    r = np.asarray(score_roman, dtype=np.float64).ravel()
    return float(np.mean(np.abs(a - r)))


def stochastic_stability(scores_per_seed: np.ndarray) -> float:
    """Average per-item std-dev across N seeds.

    Args:
        scores_per_seed: shape (n_items, n_seeds), one row per item.
    """
    arr = np.asarray(scores_per_seed, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("scores_per_seed must be 2-D with ≥2 seeds")
    return float(arr.std(axis=1, ddof=1).mean())


def length_bias_partial(
    scores: np.ndarray,
    lengths: np.ndarray,
    gold: np.ndarray,
) -> float:
    """Length-bias *partial* correlation: how much score depends on length
    after partialing out the true gold score. Lower magnitude = better.
    """
    from scipy import stats

    s = np.asarray(scores, dtype=np.float64).ravel()
    L = np.asarray(lengths, dtype=np.float64).ravel()
    g = np.asarray(gold, dtype=np.float64).ravel()
    # Residualize s and L against g, then correlate residuals.
    bs = np.polyfit(g, s, 1)
    bL = np.polyfit(g, L, 1)
    res_s = s - (bs[0] * g + bs[1])
    res_L = L - (bL[0] * g + bL[1])
    r, _ = stats.spearmanr(res_s, res_L)
    return float(r)
