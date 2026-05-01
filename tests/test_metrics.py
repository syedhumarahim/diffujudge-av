from __future__ import annotations

import numpy as np

from diffujudge.metrics import (
    cohen_kappa,
    expected_calibration_error,
    krippendorff_alpha,
    pearson,
    position_bias_delta,
    spearman,
    stochastic_stability,
)


def test_kappa_perfect_agreement():
    a = np.array([1, 2, 3, 4, 5, 5, 4, 3])
    assert cohen_kappa(a, a) == 1.0


def test_pearson_spearman_identity():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    assert pearson(x, x) > 0.999
    assert spearman(x, x) > 0.999


def test_ece_zero_for_perfect_calibration():
    p = np.linspace(1.0, 5.0, 100)
    y = p.copy()
    assert expected_calibration_error(p, y, n_bins=10) < 1e-9


def test_ece_nonzero_for_overconfidence():
    p = np.linspace(1.0, 5.0, 100)
    y = p + 0.5
    ece = expected_calibration_error(p, y, n_bins=10)
    assert ece > 0


def test_position_bias_zero_when_symmetric():
    rng = np.random.default_rng(0)
    a = rng.uniform(1, 5, size=50)
    assert position_bias_delta(a, a) == 0.0


def test_stochastic_stability_increases_with_seed_noise():
    rng = np.random.default_rng(0)
    base = rng.uniform(1, 5, size=20)
    low = base[:, None] + rng.normal(0, 0.05, size=(20, 5))
    hi = base[:, None] + rng.normal(0, 0.5, size=(20, 5))
    assert stochastic_stability(low) < stochastic_stability(hi)


def test_krippendorff_alpha_handles_two_raters():
    data = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=float)
    a = krippendorff_alpha(data, level="ordinal")
    assert a > 0.95
