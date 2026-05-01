from __future__ import annotations

import numpy as np

from diffujudge.conformal import OrdinalBoundaryConformal


def test_split_conformal_marginal_coverage():
    """Marginal coverage on a held-out test split should be ≥ 1 − α with high
    probability when the calibration and test residuals are exchangeable.
    """
    rng = np.random.default_rng(0)
    n_cal, n_test = 200, 200
    cal_pred = rng.uniform(1, 5, size=n_cal)
    cal_lab = cal_pred + rng.normal(0, 0.4, size=n_cal)
    cal_sig = np.full(n_cal, 0.4)

    test_pred = rng.uniform(1, 5, size=n_test)
    test_lab = test_pred + rng.normal(0, 0.4, size=n_test)
    test_sig = np.full(n_test, 0.4)
    test_ids = [f"t{i}" for i in range(n_test)]

    conf = OrdinalBoundaryConformal(alpha=0.10, adaptive=True, boundary_snap=False)
    conf.fit(cal_pred, cal_lab, cal_sigmas=cal_sig)
    res = conf.evaluate(test_ids, test_pred, test_lab, sigmas=test_sig)

    # Allow a small tolerance for finite-sample noise.
    assert res.coverage >= 0.85
    assert res.mean_width > 0


def test_alpha_monotonicity():
    rng = np.random.default_rng(1)
    cal_pred = rng.uniform(1, 5, size=200)
    cal_lab = cal_pred + rng.normal(0, 0.4, size=200)
    cal_sig = np.full(200, 0.4)

    widths = []
    for alpha in [0.30, 0.10, 0.01]:
        c = OrdinalBoundaryConformal(alpha=alpha, adaptive=True, boundary_snap=False)
        c.fit(cal_pred, cal_lab, cal_sigmas=cal_sig)
        ivs = c.predict([f"t{i}" for i in range(20)], cal_pred[:20], sigmas=cal_sig[:20])
        widths.append(np.mean([iv.width for iv in ivs]))
    # Tighter α → wider intervals.
    assert widths[0] < widths[1] < widths[2]
