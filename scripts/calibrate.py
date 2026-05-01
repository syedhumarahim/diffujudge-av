"""Standalone conformal calibration on a pipeline output dir.

Re-fits the OrdinalBoundaryConformal layer (or a MAPIE wrapper, if installed)
on the calibration split and reports coverage / mean width on the held-out
test split. Useful as an ablation between adaptive vs. plain split-conformal,
or for sweeping α.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import orjson

from diffujudge.conformal import OrdinalBoundaryConformal


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--cal-frac", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-adaptive", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    rows = [orjson.loads(line) for line in open(run_dir / "summary.jsonl", "rb") if line.strip()]
    rows = [r for r in rows if r.get("gold") is not None]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(rows)
    n_cal = max(int(round(args.cal_frac * len(rows))), 10)
    cal, test = rows[:n_cal], rows[n_cal:]

    cal_pred = np.array([r["point_estimate"] for r in cal])
    cal_lab = np.array([float(r["gold"]) for r in cal])
    cal_sig = np.array([np.sqrt(r["posterior_var"]) for r in cal])

    test_pred = np.array([r["point_estimate"] for r in test])
    test_lab = np.array([float(r["gold"]) for r in test])
    test_sig = np.array([np.sqrt(r["posterior_var"]) for r in test])
    test_ids = [r["item_id"] for r in test]

    conf = OrdinalBoundaryConformal(alpha=args.alpha, adaptive=not args.no_adaptive)
    conf.fit(cal_pred, cal_lab, cal_sigmas=cal_sig)
    res = conf.evaluate(test_ids, test_pred, test_lab, sigmas=test_sig)

    out = {
        "alpha": args.alpha,
        "n_cal": len(cal),
        "n_test": len(test),
        "coverage": res.coverage,
        "mean_width": res.mean_width,
        "quantile": res.quantile,
        "adaptive": not args.no_adaptive,
    }
    print(json.dumps(out, indent=2))
    (run_dir / "conformal_report.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
