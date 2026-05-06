"""Subsample existing judge outputs to simulate fewer judges / fewer samples per level.

Tests whether Tweedie helps when you have fewer observations per item.
Reads judge_outputs.jsonl, subsamples, re-runs Tweedie, and reports metrics.

Configurations tested:
  A. 3 judges × 3 samples (original) = 66 obs/item  → baseline
  B. 1 judge × 3 samples = 22 obs/item
  C. 3 judges × 1 sample = ~21 obs/item
  D. 1 judge × 1 sample = ~8 obs/item               → most room for Tweedie
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import orjson
from scipy.stats import pearsonr, spearmanr

from diffujudge.denoiser.tweedie import AnalyticalTweedieDenoiser
from diffujudge.metrics.calibration import expected_calibration_error, brier_score


def load_outputs(run_dir: Path):
    records = []
    for line in open(run_dir / "judge_outputs.jsonl", "rb"):
        if not line.strip():
            continue
        records.append(orjson.loads(line))
    return records


def load_gold(run_dir: Path) -> dict[str, float]:
    gold = {}
    for line in open(run_dir / "summary.jsonl", "rb"):
        if not line.strip():
            continue
        r = orjson.loads(line)
        if r.get("gold") is not None:
            gold[r["item_id"]] = float(r["gold"])
    return gold


def parse_level(sample_id: str) -> int:
    try:
        return int(sample_id.split("::t")[1].split("::")[0])
    except (IndexError, ValueError):
        return 0


def parse_k(sample_id: str) -> int:
    try:
        return int(sample_id.split("::k")[1])
    except (IndexError, ValueError):
        return 0


def subsample(records, max_judges: int | None = None, max_k: int | None = None):
    """Subsample records keeping at most `max_judges` judges and `max_k` samples per level."""
    judges = sorted(set(r["judge"] for r in records))
    if max_judges is not None:
        judges = judges[:max_judges]
    judge_set = set(judges)

    filtered = []
    seen = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in records:
        if r["judge"] not in judge_set:
            continue
        level = parse_level(r["sample_id"])
        key = (r["item_id"], level, r["judge"])
        if max_k is not None and seen[key][r["judge"]] >= max_k:
            continue
        seen[key][r["judge"]] += 1
        filtered.append(r)
    return filtered


def evaluate(records, gold: dict[str, float], label: str):
    by_item: dict[str, list[float]] = collections.defaultdict(list)
    levels_by_item: dict[str, list[int]] = collections.defaultdict(list)
    for r in records:
        by_item[r["item_id"]].append(float(r["score"]))
        levels_by_item[r["item_id"]].append(parse_level(r["sample_id"]))

    den = AnalyticalTweedieDenoiser(bandwidth="scott", precision_weight=False)

    items_with_gold = [iid for iid in by_item if iid in gold]
    if not items_with_gold:
        return None

    gold_arr = np.array([gold[iid] for iid in items_with_gold])
    raw_arr = np.array([np.mean(by_item[iid]) for iid in items_with_gold])

    den_arr = []
    post_std_arr = []
    for iid in items_with_gold:
        scores = np.array(by_item[iid], dtype=np.float64)
        levels = np.array(levels_by_item[iid], dtype=np.int64)
        est = den.denoise_item(iid, scores, levels)
        den_arr.append(est.point_estimate)
        post_std_arr.append(est.posterior_std)
    den_arr = np.array(den_arr)
    post_std_arr = np.array(post_std_arr)

    pr_raw = float(pearsonr(gold_arr, raw_arr)[0])
    pr_den = float(pearsonr(gold_arr, den_arr)[0])
    sp_raw = float(spearmanr(gold_arr, raw_arr)[0])
    sp_den = float(spearmanr(gold_arr, den_arr)[0])
    ece_raw = float(expected_calibration_error(raw_arr, gold_arr, score_min=1, score_max=5))
    ece_den = float(expected_calibration_error(den_arr, gold_arr, score_min=1, score_max=5))
    brier_raw = float(brier_score(raw_arr, gold_arr))
    brier_den = float(brier_score(den_arr, gold_arr))

    n_per_item = np.mean([len(by_item[iid]) for iid in items_with_gold])
    mean_post_std = float(post_std_arr.mean())

    return {
        "label": label,
        "n_items": len(items_with_gold),
        "n_records": len(records),
        "obs_per_item": round(n_per_item, 1),
        "pearson_raw": pr_raw,
        "pearson_denoised": pr_den,
        "pearson_delta": pr_den - pr_raw,
        "spearman_raw": sp_raw,
        "spearman_denoised": sp_den,
        "spearman_delta": sp_den - sp_raw,
        "ece_raw": ece_raw,
        "ece_denoised": ece_den,
        "ece_delta": ece_den - ece_raw,
        "brier_raw": brier_raw,
        "brier_denoised": brier_den,
        "mean_posterior_std": mean_post_std,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="outputs/4f66efea222c4b39")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    records = load_outputs(run_dir)
    gold = load_gold(run_dir)

    configs = [
        ("A: 3 judges × full k (original)", None, None),
        ("B: 1 judge × full k", 1, None),
        ("C: 3 judges × k=1", None, 1),
        ("D: 1 judge × k=1", 1, 1),
    ]

    results = []
    print("=" * 80)
    print("SUBSAMPLE EXPERIMENT: Does Tweedie help with fewer observations?")
    print("=" * 80)

    for label, max_j, max_k in configs:
        sub = subsample(records, max_judges=max_j, max_k=max_k)
        m = evaluate(sub, gold, label)
        if m is None:
            continue
        results.append(m)

        delta_p = m["pearson_delta"]
        delta_s = m["spearman_delta"]
        delta_e = m["ece_delta"]
        marker = "✓" if delta_p > 0.005 else "≈" if abs(delta_p) <= 0.005 else "✗"

        print(f"\n  [{label}]  ({m['obs_per_item']} obs/item, {m['n_records']} records)")
        print(f"    Pearson:  {m['pearson_raw']:.4f} → {m['pearson_denoised']:.4f}  (Δ = {delta_p:+.4f}) {marker}")
        print(f"    Spearman: {m['spearman_raw']:.4f} → {m['spearman_denoised']:.4f}  (Δ = {delta_s:+.4f})")
        print(f"    ECE:      {m['ece_raw']:.4f} → {m['ece_denoised']:.4f}  (Δ = {delta_e:+.4f})")
        print(f"    Post σ̂:  {m['mean_posterior_std']:.4f}")

    # Also try with precision weighting on the sparse configs
    print("\n" + "-" * 80)
    print("WITH PRECISION WEIGHTING (helps when levels have different noise)")
    print("-" * 80)

    for label, max_j, max_k in configs[2:]:  # only C and D
        sub = subsample(records, max_judges=max_j, max_k=max_k)

        by_item = collections.defaultdict(list)
        levels_by_item = collections.defaultdict(list)
        for r in sub:
            by_item[r["item_id"]].append(float(r["score"]))
            levels_by_item[r["item_id"]].append(parse_level(r["sample_id"]))

        den = AnalyticalTweedieDenoiser(bandwidth="scott", precision_weight=True)
        items_with_gold = [iid for iid in by_item if iid in gold]
        gold_arr = np.array([gold[iid] for iid in items_with_gold])
        raw_arr = np.array([np.mean(by_item[iid]) for iid in items_with_gold])

        den_arr = []
        for iid in items_with_gold:
            scores = np.array(by_item[iid], dtype=np.float64)
            levels = np.array(levels_by_item[iid], dtype=np.int64)
            est = den.denoise_item(iid, scores, levels)
            den_arr.append(est.point_estimate)
        den_arr = np.array(den_arr)

        pr_raw = float(pearsonr(gold_arr, raw_arr)[0])
        pr_den = float(pearsonr(gold_arr, den_arr)[0])
        delta_p = pr_den - pr_raw
        marker = "✓" if delta_p > 0.005 else "≈" if abs(delta_p) <= 0.005 else "✗"

        print(f"\n  [{label} + precision_weight=True]")
        print(f"    Pearson:  {pr_raw:.4f} → {pr_den:.4f}  (Δ = {delta_p:+.4f}) {marker}")

    out_path = Path("docs/figures/subsample_experiment.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[ok] results → {out_path}")


if __name__ == "__main__":
    main()
