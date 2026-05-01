"""Inter-rater reliability and rank-correlation metrics for judge evaluation.

These are the standard metrics in the LLM-as-a-Judge literature
(MT-Bench, Chatbot Arena, BiGGen-Bench, Lingo-Judge).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score


def cohen_kappa(rater_a: np.ndarray, rater_b: np.ndarray, weights: str | None = "quadratic") -> float:
    """Cohen's κ. Use weights='quadratic' for ordinal scales (default)."""
    return float(cohen_kappa_score(rater_a, rater_b, weights=weights))


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss's κ for ≥3 raters on the same items.

    Args:
        ratings: shape (n_items, n_categories), entries are counts of how many
                 raters assigned each category to that item.
    """
    ratings = np.asarray(ratings, dtype=np.float64)
    n_items, n_cats = ratings.shape
    n_raters = ratings.sum(axis=1)
    if not np.allclose(n_raters, n_raters[0]):
        raise ValueError("Fleiss's κ requires the same number of raters per item")
    n = int(n_raters[0])
    if n < 2:
        raise ValueError("Need ≥2 raters per item")
    p_j = ratings.sum(axis=0) / (n_items * n)
    P_i = ((ratings**2).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = (p_j**2).sum()
    if 1.0 - P_e < 1e-12:
        return 1.0
    return float((P_bar - P_e) / (1.0 - P_e))


def krippendorff_alpha(
    ratings: np.ndarray,
    level: str = "ordinal",
) -> float:
    """Krippendorff's α (handles missing values; supports ordinal scales).

    Falls back to a numpy implementation if the `krippendorff` package is
    unavailable. Pass `np.nan` for missing observations.
    """
    try:
        import krippendorff as _kr

        return float(_kr.alpha(reliability_data=ratings, level_of_measurement=level))
    except ImportError:  # pragma: no cover
        return _krippendorff_alpha_numpy(ratings, level=level)


def _krippendorff_alpha_numpy(ratings: np.ndarray, level: str = "ordinal") -> float:
    """Numpy fallback (interval/ordinal) — not as fast as the PyPI package, but correct.

    Reference: Krippendorff (2011), 'Computing Krippendorff's α-Reliability'.
    """
    R = np.asarray(ratings, dtype=np.float64)
    pairs: list[tuple[float, float]] = []
    for col in range(R.shape[1]):
        values = R[:, col]
        values = values[~np.isnan(values)]
        if values.size < 2:
            continue
        for i in range(values.size):
            for j in range(values.size):
                if i != j:
                    pairs.append((values[i], values[j]))
    if not pairs:
        return float("nan")
    arr = np.array(pairs)
    if level == "interval":
        d_obs = float(np.mean((arr[:, 0] - arr[:, 1]) ** 2))
        all_vals = R[~np.isnan(R)]
        m = all_vals.mean()
        d_exp = float(2 * np.mean((all_vals - m) ** 2))
    else:  # ordinal — rank-based metric difference
        all_vals = R[~np.isnan(R)]
        ranks = stats.rankdata(all_vals)
        rank_map = {v: r for v, r in zip(all_vals, ranks)}
        a = np.array([rank_map[x] for x in arr[:, 0]])
        b = np.array([rank_map[x] for x in arr[:, 1]])
        d_obs = float(np.mean((a - b) ** 2))
        m = ranks.mean()
        d_exp = float(2 * np.mean((ranks - m) ** 2))
    if d_exp <= 0:
        return 1.0
    return 1.0 - d_obs / d_exp


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    r, _ = stats.pearsonr(a, b)
    return float(r)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    r, _ = stats.spearmanr(a, b)
    return float(r)


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    t, _ = stats.kendalltau(a, b)
    return float(t)
