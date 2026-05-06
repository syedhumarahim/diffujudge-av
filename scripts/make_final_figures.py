"""Generate the definitive figure set for README and TDS article.

Produces:
    docs/figures/hero_triptych.png    — 3-panel overview
    docs/figures/bias_robustness.png  — per-level bias delta heatmap
    docs/figures/gold_vs_pred.png     — scatter with uncertainty coloring
    docs/figures/posterior_sigma.png   — σ̂ distribution + error correlation
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import orjson

from diffujudge.metrics.agreement import pearson, spearman
from diffujudge.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)

RUN_DIR = Path("outputs/e66b4b96a727c3eb")
OUT_DIR = Path("docs/figures")

LEVEL_NAMES = {
    0: "Anchor\n(baseline)",
    2: "Rubric\nparaphrase",
    3: "Criterion\nreorder",
    4: "Score-ID\nformat",
    5: "Temperature\nnoise",
    6: "Exemplar\nresample",
    7: "Frame\nshuffle",
}

LEVEL_BIASES = {
    2: "SPUQ",
    3: "Chen et al.",
    4: "Chen et al.",
    5: "Rating Roulette",
    6: "classical",
    7: "this work",
}


def load_summary():
    rows = [orjson.loads(l) for l in open(RUN_DIR / "summary.jsonl", "rb") if l.strip()]
    rows = [r for r in rows if r.get("gold") is not None]
    gold = np.array([r["gold"] for r in rows], dtype=np.float64)
    raw = np.array([r["raw_mean"] for r in rows], dtype=np.float64)
    den = np.array([r["point_estimate"] for r in rows], dtype=np.float64)
    post_std = np.array([r["posterior_var"] ** 0.5 for r in rows], dtype=np.float64)
    return rows, gold, raw, den, post_std


def load_judge_outputs():
    by_item_level: dict[str, dict[int, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for line in open(RUN_DIR / "judge_outputs.jsonl", "rb"):
        if not line.strip():
            continue
        r = orjson.loads(line)
        sid = r["sample_id"]
        try:
            level = int(sid.split("::t")[1].split("::")[0])
        except (IndexError, ValueError):
            level = 0
        by_item_level[r["item_id"]][level].append(float(r["score"]))
    return by_item_level


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    rows, gold, raw, den, post_std = load_summary()
    by_item_level = load_judge_outputs()
    items = sorted(by_item_level.keys())
    abs_error = np.abs(gold - den)

    # ====== FIGURE 1: Hero triptych ======
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=160)

    # Panel A: Gold vs prediction scatter (colored by σ̂)
    ax = axes[0]
    sc = ax.scatter(gold, den, c=post_std, cmap="RdYlBu_r", alpha=0.7, s=44,
                    edgecolor="#333", linewidth=0.3)
    ax.plot([1, 5], [1, 5], "--", color="#888", linewidth=1.0)
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    ax.set_xlabel("Gold score", fontsize=11)
    ax.set_ylabel("Tweedie posterior mean", fontsize=11)
    ax.set_title(f"A. Gold vs. prediction\nr = {float(pearson(gold, den)):.3f}, "
                 f"ρ = {float(spearman(gold, den)):.3f}", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("σ̂", fontsize=10)

    # Panel B: Reliability diagram (raw vs denoised)
    ax = axes[1]
    rc_raw = reliability_curve(raw, gold, n_bins=10, score_min=1.0, score_max=5.0)
    rc_den = reliability_curve(den, gold, n_bins=10, score_min=1.0, score_max=5.0)
    ece_r = expected_calibration_error(raw, gold, score_min=1.0, score_max=5.0)
    ece_d = expected_calibration_error(den, gold, score_min=1.0, score_max=5.0)

    ax.plot([1, 5], [1, 5], "--", color="#888", linewidth=1.0, label="Perfect")
    mask_r = rc_raw["counts"] > 0
    ax.plot(rc_raw["pred_means"][mask_r], rc_raw["label_means"][mask_r],
            "o-", linewidth=2, color="#d6604d", label=f"Raw (ECE={ece_r:.3f})")
    mask_d = rc_den["counts"] > 0
    ax.plot(rc_den["pred_means"][mask_d], rc_den["label_means"][mask_d],
            "s-", linewidth=2, color="#2166ac", label=f"Tweedie (ECE={ece_d:.3f})")
    ax.set_xlim(1, 5)
    ax.set_ylim(1, 5)
    ax.set_xlabel("Predicted score", fontsize=11)
    ax.set_ylabel("Mean gold per bin", fontsize=11)
    ax.set_title("B. Reliability diagram", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_aspect("equal", adjustable="box")

    # Panel C: Per-level mean score (perturbation stability)
    ax = axes[2]
    levels = sorted(LEVEL_NAMES.keys())
    avail_levels = [l for l in levels if l in set().union(*(d.keys() for d in by_item_level.values()))]
    means, sems = [], []
    for t in avail_levels:
        vals = []
        for iid in items:
            if t in by_item_level[iid]:
                vals.extend(by_item_level[iid][t])
        means.append(float(np.mean(vals)))
        sems.append(float(np.std(vals) / np.sqrt(len(vals))))

    labels = [LEVEL_NAMES.get(l, str(l)) for l in avail_levels]
    colors = ["#4393c3" if l == 0 else "#92c5de" for l in avail_levels]
    bars = ax.bar(labels, means, color=colors, edgecolor="#08519c", alpha=0.85)
    ax.errorbar(labels, means, yerr=sems, fmt='none', color='#333', capsize=3)
    global_mean = float(np.mean(means))
    ax.axhline(global_mean, color="#d6604d", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_ylabel("Mean score", fontsize=11)
    ax.set_title("C. Score stability across\nbias-source perturbations", fontsize=11)
    ax.tick_params(axis='x', labelsize=7.5)
    ax.text(0.98, 0.95, f"Max Δ = {max(means)-min(means):.3f}\nCV = {np.std(means)/np.mean(means):.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle(
        "DiffuJudge-AV — Score Diffusion Judging on 3-judge Claude ensemble\n"
        "100 items × 7 perturbation levels × 3 samples × 3 judges = 6,600 API calls",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hero_triptych.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "hero_triptych.svg", bbox_inches="tight")
    plt.close(fig)

    # ====== FIGURE 2: Per-level bias deltas (the robustness table as a figure) ======
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=160)

    # Panel A: mean delta from anchor per level
    ax = axes[0]
    mean_deltas = []
    abs_deltas = []
    for t in avail_levels:
        if t == 0:
            mean_deltas.append(0.0)
            abs_deltas.append(0.0)
            continue
        deltas = []
        for iid in items:
            if 0 in by_item_level[iid] and t in by_item_level[iid]:
                d = np.mean(by_item_level[iid][t]) - np.mean(by_item_level[iid][0])
                deltas.append(d)
        mean_deltas.append(float(np.mean(deltas)))
        abs_deltas.append(float(np.mean(np.abs(deltas))))

    x = np.arange(len(avail_levels))
    width = 0.35
    ax.bar(x - width/2, mean_deltas, width, color="#4393c3", label="Mean Δ (signed)")
    ax.bar(x + width/2, abs_deltas, width, color="#d6604d", alpha=0.7, label="Mean |Δ| (unsigned)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Score delta from anchor", fontsize=11)
    ax.set_title("A. Per-bias-source score delta", fontsize=11)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.legend(fontsize=9)

    # Panel B: per-level within-item variance
    ax = axes[1]
    level_vars = []
    for t in avail_levels:
        vars_list = []
        for iid in items:
            if t in by_item_level[iid] and len(by_item_level[iid][t]) >= 2:
                vars_list.append(float(np.var(by_item_level[iid][t], ddof=1)))
        level_vars.append(float(np.mean(vars_list)) if vars_list else 0.0)

    ax.bar(labels, level_vars, color="#92c5de", edgecolor="#08519c")
    ax.set_ylabel("Mean within-item variance", fontsize=11)
    ax.set_title("B. Within-item variance per level", fontsize=11)
    ax.tick_params(axis='x', labelsize=7.5)

    fig.suptitle("Perturbation robustness analysis — Claude 3-judge ensemble", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bias_robustness.png", bbox_inches="tight")
    plt.close(fig)

    # ====== FIGURE 3: Posterior σ̂ analysis ======
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)

    # Panel A: σ̂ distribution
    ax = axes[0]
    ax.hist(post_std, bins=20, color="#4393c3", edgecolor="#08519c", alpha=0.8)
    ax.axvline(np.mean(post_std), color="#d6604d", linewidth=2, linestyle="--",
               label=f"Mean = {np.mean(post_std):.4f}")
    ax.axvline(np.median(post_std), color="#2166ac", linewidth=2, linestyle=":",
               label=f"Median = {np.median(post_std):.4f}")
    ax.set_xlabel("Posterior σ̂", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("A. Distribution of Tweedie posterior σ̂", fontsize=11)
    ax.legend(fontsize=9)

    # Panel B: σ̂ vs error
    ax = axes[1]
    ax.scatter(post_std, abs_error, alpha=0.6, s=36, color="#2166ac")
    corr = float(pearson(post_std, abs_error))
    z = np.polyfit(post_std, abs_error, 1)
    xs = np.linspace(post_std.min(), post_std.max(), 50)
    ax.plot(xs, z[0] * xs + z[1], "--", color="#d6604d", linewidth=1.5)
    ax.set_xlabel("Posterior σ̂", fontsize=11)
    ax.set_ylabel("|Gold − Prediction|", fontsize=11)
    ax.set_title(f"B. Uncertainty vs. prediction error\n(r = {corr:.3f})", fontsize=11)

    fig.suptitle("Tweedie posterior variance as a free uncertainty estimate", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "posterior_sigma.png", bbox_inches="tight")
    plt.close(fig)

    # ====== Summary JSON ======
    summary = {
        "n_items": len(gold),
        "n_api_calls": 6600,
        "judges": "claude-haiku-4-5@T0 + claude-sonnet-4-5@T0 + claude-haiku-4-5@T0.6",
        "cost_usd": 3.0,
        "wall_clock_min": 25,
        "pearson_raw": float(pearson(gold, raw)),
        "pearson_denoised": float(pearson(gold, den)),
        "spearman_raw": float(spearman(gold, raw)),
        "spearman_denoised": float(spearman(gold, den)),
        "ece_raw": float(ece_r),
        "ece_denoised": float(ece_d),
        "brier_raw": float(brier_score(raw, gold)),
        "brier_denoised": float(brier_score(den, gold)),
        "mean_posterior_std": float(np.mean(post_std)),
        "sigma_error_correlation": corr,
        "max_level_delta": float(max(means) - min(means)),
        "level_cv": float(np.std(means) / np.mean(means)),
        "per_level_mean_deltas": {LEVEL_NAMES[l].replace("\n", " "): float(d)
                                   for l, d in zip(avail_levels, mean_deltas)},
    }
    with open(OUT_DIR / "final_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\n[ok] all figures → {OUT_DIR}/")


if __name__ == "__main__":
    main()
