"""Single-step Tweedie posterior-mean denoiser for LLM-judge scores.

Background
----------
Tweedie's identity (Robbins 1956; Efron 2011): for y = x + ε with ε ~ N(0, σ²),
the posterior mean is

    E[x | y] = y + σ² · ∇_y log p(y)

where p(y) is the marginal density of the noisy observations. The
second-order extension (Manor & Michaeli, ICLR 2024) gives the posterior
variance for free:

    Var[x | y] = σ² + σ⁴ · ∇²_y log p(y)

We estimate p(y) with a Gaussian KDE over the N×k perturbed-score samples per
item. With bandwidth h, the score function of a Gaussian KDE has a clean
softmax-weighted-residual closed form, so this is training-free and runs in
~milliseconds per item.

Reference framing
-----------------
- Manor & Michaeli, *Posterior-Mean Denoising via Tweedie's Formula*, ICLR 2024
  (arXiv 2309.13598).
- Gao et al., *SPUQ — Perturbation-based Uncertainty Quantification for LLMs*
  (arXiv 2403.02509), which motivates per-level σ_t estimation.

Per-level σ_t estimation
------------------------
We bucket samples by their forward-process level t and use the within-bucket
variance as σ_t² — this aligns with the "known-noise-schedule" framing where
each level corresponds to one bias source. Pool across levels with precision
weighting before applying the Tweedie correction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TweedieEstimate:
    """One item's denoised judgment."""

    item_id: str
    point_estimate: float        # ŝ_0 — Tweedie posterior mean
    posterior_var: float          # σ̂² — posterior variance (Manor & Michaeli)
    posterior_std: float
    raw_mean: float               # baseline: simple ensemble mean
    raw_std: float
    n_samples: int
    sigma_per_level: dict[int, float]
    level_means: dict[int, float]


def estimate_per_level_sigma(
    scores: np.ndarray,
    levels: np.ndarray,
    floor: float = 1e-3,
) -> dict[int, float]:
    """Estimate σ_t per perturbation level as the within-bucket std-dev.

    Levels with a single observation fall back to the global std. We floor
    σ_t to avoid degenerate precision weights that would let one outlier
    dominate the pool.
    """
    out: dict[int, float] = {}
    global_std = float(np.std(scores)) if scores.size > 1 else 1.0
    for t in np.unique(levels):
        bucket = scores[levels == t]
        if bucket.size >= 2:
            sigma = float(np.std(bucket, ddof=1))
        else:
            sigma = global_std
        out[int(t)] = max(sigma, floor)
    return out


def _kde_log_density_score(
    y: np.ndarray,
    samples: np.ndarray,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (∇log p̂(y), ∇²log p̂(y)) under a 1-D Gaussian KDE with bandwidth h.

    Closed form:
        let w_i(y) = softmax_i(-(y - y_i)² / (2h²))
        ∇log p̂(y)   = (1/h²) Σ_i w_i(y) · (y_i - y)
                    = (1/h²) (μ_w(y) - y)
        ∇²log p̂(y) = (1/h⁴) Var_w(y_i) - 1/h²

    where μ_w, Var_w are the softmax-weighted moments of {y_i}.
    """
    y = np.atleast_1d(y).astype(np.float64)                 # (M,)
    samples = samples.astype(np.float64)                     # (N,)
    diffs = samples[None, :] - y[:, None]                    # (M, N)
    log_w = -0.5 * (diffs / h) ** 2
    log_w -= log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    w /= w.sum(axis=1, keepdims=True)                        # (M, N)

    weighted_mean = (w * samples[None, :]).sum(axis=1)       # μ_w(y)
    weighted_sq = (w * samples[None, :] ** 2).sum(axis=1)
    weighted_var = np.maximum(weighted_sq - weighted_mean**2, 0.0)

    grad = (weighted_mean - y) / (h**2)
    hess = weighted_var / (h**4) - 1.0 / (h**2)
    return grad, hess


def _scott_bandwidth(samples: np.ndarray) -> float:
    n = max(samples.size, 2)
    std = float(np.std(samples, ddof=1))
    if std <= 0:
        std = 1e-2
    return std * n ** (-1.0 / 5.0)


def _silverman_bandwidth(samples: np.ndarray) -> float:
    n = max(samples.size, 2)
    std = float(np.std(samples, ddof=1))
    if std <= 0:
        std = 1e-2
    iqr = float(np.subtract(*np.percentile(samples, [75, 25])))
    a = min(std, iqr / 1.34) if iqr > 0 else std
    return 0.9 * a * n ** (-1.0 / 5.0)


class AnalyticalTweedieDenoiser:
    """Training-free Tweedie denoiser via Gaussian-KDE score estimation.

    Workflow per item:
      1. Pool N×k perturbed samples; compute precision-weighted mean ȳ
         using per-level σ_t² as inverse weights.
      2. Build a Gaussian KDE over all samples.
      3. Compute Tweedie correction at y = ȳ:
            ŝ_0 = ȳ + σ̂_ε² · ∇log p̂(ȳ)
         where σ̂_ε² is the precision-weighted noise variance.
      4. Posterior variance from Manor & Michaeli's second-order identity.
      5. Clip to the legal score interval.
    """

    def __init__(
        self,
        score_min: float = 1.0,
        score_max: float = 5.0,
        bandwidth: float | str = "scott",
        precision_weight: bool = True,
    ) -> None:
        self.score_min = float(score_min)
        self.score_max = float(score_max)
        self.bandwidth = bandwidth
        self.precision_weight = precision_weight

    def _resolve_bandwidth(self, samples: np.ndarray) -> float:
        if isinstance(self.bandwidth, (int, float)):
            return max(float(self.bandwidth), 1e-3)
        if self.bandwidth == "silverman":
            return max(_silverman_bandwidth(samples), 1e-3)
        return max(_scott_bandwidth(samples), 1e-3)  # default: Scott's rule

    def denoise_item(
        self,
        item_id: str,
        scores: np.ndarray,
        levels: np.ndarray,
    ) -> TweedieEstimate:
        scores = np.asarray(scores, dtype=np.float64).ravel()
        levels = np.asarray(levels, dtype=np.int64).ravel()
        if scores.size == 0:
            raise ValueError(f"No samples for item {item_id}")
        if scores.size != levels.size:
            raise ValueError("scores and levels length mismatch")

        sigma_per_level = estimate_per_level_sigma(scores, levels)
        level_means = {
            int(t): float(scores[levels == t].mean()) for t in np.unique(levels)
        }

        if self.precision_weight and len(sigma_per_level) > 1:
            inv_var = np.array([1.0 / sigma_per_level[int(t)] ** 2 for t in levels])
            weights = inv_var / inv_var.sum()
            y_bar = float((scores * weights).sum())
            sigma_eps_sq = 1.0 / inv_var.sum()
        else:
            y_bar = float(scores.mean())
            sigma_eps_sq = float(np.var(scores, ddof=1)) / max(scores.size, 1)

        h = self._resolve_bandwidth(scores)
        grad, hess = _kde_log_density_score(np.array([y_bar]), scores, h)
        grad_v = float(grad[0])
        hess_v = float(hess[0])

        point = y_bar + sigma_eps_sq * grad_v
        post_var = sigma_eps_sq + (sigma_eps_sq**2) * hess_v
        post_var = max(post_var, 1e-6)

        point = float(np.clip(point, self.score_min, self.score_max))

        raw_mean = float(scores.mean())
        raw_std = float(scores.std(ddof=1)) if scores.size > 1 else 0.0

        return TweedieEstimate(
            item_id=item_id,
            point_estimate=point,
            posterior_var=post_var,
            posterior_std=float(np.sqrt(post_var)),
            raw_mean=raw_mean,
            raw_std=raw_std,
            n_samples=int(scores.size),
            sigma_per_level={int(k): float(v) for k, v in sigma_per_level.items()},
            level_means=level_means,
        )

    def denoise_batch(
        self,
        item_ids: list[str],
        scores_per_item: list[np.ndarray],
        levels_per_item: list[np.ndarray],
    ) -> list[TweedieEstimate]:
        return [
            self.denoise_item(iid, s, l)
            for iid, s, l in zip(item_ids, scores_per_item, levels_per_item, strict=True)
        ]
