from __future__ import annotations

import numpy as np

from diffujudge.denoiser import AnalyticalTweedieDenoiser, estimate_per_level_sigma


def test_per_level_sigma_within_bucket():
    scores = np.array([3.0, 3.1, 2.9, 4.0, 4.2, 3.8])
    levels = np.array([0, 0, 0, 1, 1, 1])
    out = estimate_per_level_sigma(scores, levels)
    assert set(out) == {0, 1}
    assert all(v > 0 for v in out.values())


def test_tweedie_recovers_unimodal_mean():
    """When samples concentrate around a true mean μ with light Gaussian noise,
    the Tweedie posterior mean should be very close to μ.
    """
    rng = np.random.default_rng(0)
    mu = 3.7
    scores = rng.normal(mu, 0.2, size=60)
    levels = np.zeros_like(scores, dtype=int)
    den = AnalyticalTweedieDenoiser(score_min=1.0, score_max=5.0, bandwidth="scott")
    est = den.denoise_item("x", scores, levels)
    assert abs(est.point_estimate - mu) < 0.15
    assert est.posterior_var > 0


def test_tweedie_clips_to_range():
    rng = np.random.default_rng(0)
    scores = rng.normal(0.5, 0.5, size=30)
    levels = np.zeros_like(scores, dtype=int)
    den = AnalyticalTweedieDenoiser(score_min=1.0, score_max=5.0)
    est = den.denoise_item("x", scores, levels)
    assert 1.0 <= est.point_estimate <= 5.0


def test_precision_weighting_pulls_toward_low_sigma_bucket():
    """If level 0 has tight std and level 1 has loose std, the precision-weighted
    mean should sit closer to the level-0 mean than the unweighted mean would.
    """
    rng = np.random.default_rng(0)
    s0 = rng.normal(3.0, 0.05, size=30)
    s1 = rng.normal(4.5, 0.6, size=30)
    scores = np.concatenate([s0, s1])
    levels = np.concatenate([np.zeros(30, dtype=int), np.ones(30, dtype=int)])

    den = AnalyticalTweedieDenoiser(precision_weight=True)
    est = den.denoise_item("x", scores, levels)
    raw_mean = scores.mean()
    assert abs(est.point_estimate - 3.0) < abs(raw_mean - 3.0)
