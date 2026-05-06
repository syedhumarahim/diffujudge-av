"""Compute per-bias-source robustness deltas from judge_outputs.jsonl.

Reads the raw outputs, pairs observations by perturbation level, and
computes position_bias_delta, scoring_id_bias_delta, stochastic_stability,
and per-level variance. Patches the eval_report.json with the results.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import orjson


def main(run_dir: str = "outputs/e66b4b96a727c3eb") -> None:
    run_dir = Path(run_dir)
    records = [
        orjson.loads(l)
        for l in open(run_dir / "judge_outputs.jsonl", "rb")
        if l.strip()
    ]

    # Group by (item_id, level)
    by_item_level: dict[str, dict[int, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    by_item: dict[str, list[float]] = collections.defaultdict(list)

    for r in records:
        sid = r["sample_id"]
        try:
            level = int(sid.split("::t")[1].split("::")[0])
        except (IndexError, ValueError):
            level = 0
        by_item_level[r["item_id"]][level].append(float(r["score"]))
        by_item[r["item_id"]].append(float(r["score"]))

    items = sorted(by_item_level.keys())
    n = len(items)

    # Position bias: level 0 (anchor) vs level 1 (option swap)
    anchor_scores = []
    swap_scores = []
    for iid in items:
        levels = by_item_level[iid]
        if 0 in levels and 1 in levels:
            anchor_scores.append(np.mean(levels[0]))
            swap_scores.append(np.mean(levels[1]))
    pos_delta = float(np.mean(np.abs(np.array(anchor_scores) - np.array(swap_scores))))

    # Scoring-ID bias: level 0 (anchor, arabic) vs level 4 (score-ID swap)
    id_anchor = []
    id_swap = []
    for iid in items:
        levels = by_item_level[iid]
        if 0 in levels and 4 in levels:
            id_anchor.append(np.mean(levels[0]))
            id_swap.append(np.mean(levels[4]))
    sid_delta = float(np.mean(np.abs(np.array(id_anchor) - np.array(id_swap))))

    # Per-level variance (within-item, across samples)
    level_variances: dict[int, list[float]] = collections.defaultdict(list)
    for iid in items:
        for lvl, scores in by_item_level[iid].items():
            if len(scores) >= 2:
                level_variances[lvl].append(float(np.var(scores, ddof=1)))

    per_level_var = {
        lvl: float(np.mean(vars_list))
        for lvl, vars_list in sorted(level_variances.items())
    }

    # Stochastic stability: std across ALL observations per item
    stoch_stds = [float(np.std(scores)) for scores in by_item.values()]
    stoch_stability = float(np.mean(stoch_stds))

    # Per-level mean delta from anchor
    level_names = {
        0: "anchor", 1: "option_swap", 2: "rubric_paraphrase",
        3: "criterion_reorder", 4: "score_id_swap", 5: "temperature_noise",
        6: "exemplar_resample", 7: "frame_shuffle",
    }
    per_level_delta = {}
    for lvl in range(8):
        deltas = []
        for iid in items:
            if 0 in by_item_level[iid] and lvl in by_item_level[iid]:
                d = np.mean(by_item_level[iid][lvl]) - np.mean(by_item_level[iid][0])
                deltas.append(d)
        if deltas:
            per_level_delta[level_names.get(lvl, f"level_{lvl}")] = {
                "mean_delta": float(np.mean(deltas)),
                "abs_mean_delta": float(np.mean(np.abs(deltas))),
                "std_delta": float(np.std(deltas)),
            }

    results = {
        "n_items": n,
        "n_records": len(records),
        "position_bias_delta": pos_delta,
        "scoring_id_bias_delta": sid_delta,
        "stochastic_stability": stoch_stability,
        "per_level_mean_variance": per_level_var,
        "per_level_deltas": per_level_delta,
    }

    print(json.dumps(results, indent=2))

    # Patch eval_report.json
    report_path = run_dir / "eval_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        report["position_bias_delta"] = pos_delta
        report["scoring_id_bias_delta"] = sid_delta
        report["stochastic_stability"] = stoch_stability
        report["per_level_deltas"] = per_level_delta
        report_path.write_text(json.dumps(report, indent=2))
        print(f"\n[ok] patched {report_path}")


if __name__ == "__main__":
    main()
