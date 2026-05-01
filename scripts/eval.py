"""Run the eval-of-eval harness against a finished pipeline output dir."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import orjson

from diffujudge.eval import EvalOfEvalHarness


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    rows = [
        orjson.loads(line)
        for line in open(run_dir / "summary.jsonl", "rb")
        if line.strip()
    ]
    rows = [r for r in rows if r.get("gold") is not None]
    if not rows:
        raise SystemExit("No items with gold labels in summary.jsonl")

    item_ids = [r["item_id"] for r in rows]
    denoised = np.array([r["point_estimate"] for r in rows])
    raw = np.array([r["raw_mean"] for r in rows])
    gold = {r["item_id"]: float(r["gold"]) for r in rows}

    rep = EvalOfEvalHarness().run(item_ids, denoised, raw, gold)
    rep.save(run_dir / "eval_report.json")

    print(f"  n_items                 : {rep.n_items}")
    print(f"  Cohen's κ (quadratic)   : {rep.cohen_kappa:.3f}")
    print(f"  Krippendorff α          : {rep.krippendorff_alpha:.3f}")
    print(f"  Pearson r               : {rep.pearson:.3f}")
    print(f"  Spearman ρ              : {rep.spearman:.3f}")
    print(f"  Kendall τ               : {rep.kendall_tau:.3f}")
    print(f"  ECE  baseline → denoised: {rep.ece_baseline:.3f} → {rep.ece_denoised:.3f}")
    print(f"  Brier baseline → denoise: {rep.brier_baseline:.3f} → {rep.brier_denoised:.3f}")


if __name__ == "__main__":
    main()
