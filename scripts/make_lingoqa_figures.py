"""Generate figures comparing Synthetic vs LingoQA runs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import orjson
from scipy.stats import pearsonr, spearmanr

from diffujudge.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)

SYNTH_DIR = Path("outputs/e66b4b96a727c3eb")
LINGO_DIR = Path("outputs/4f66efea222c4b39")
OUT_DIR = Path("docs/figures")


def load(run_dir):
    rows = [orjson.loads(l) for l in open(run_dir / "summary.jsonl", "rb") if l.strip()]
    rows = [r for r in rows if r.get("gold") is not None]
    gold = np.array([r["gold"] for r in rows], dtype=np.float64)
    raw = np.array([r["raw_mean"] for r in rows], dtype=np.float64)
    den = np.array([r["point_estimate"] for r in rows], dtype=np.float64)
    post_std = np.array([r["posterior_var"] ** 0.5 for r in rows], dtype=np.float64)
    return rows, gold, raw, den, post_std


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    _, g_s, r_s, d_s, ps_s = load(SYNTH_DIR)
    _, g_l, r_l, d_l, ps_l = load(LINGO_DIR)

    # ====== HERO: 2×2 comparison ======
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=160)

    datasets = [
        ("Synthetic AV corpus (n=100)", g_s, r_s, d_s, ps_s),
        ("LingoQA evaluation (n=200)", g_l, r_l, d_l, ps_l),
    ]

    for row, (label, gold, raw, den, ps) in enumerate(datasets):
        pr_raw = float(pearsonr(gold, raw)[0])
        pr_den = float(pearsonr(gold, den)[0])
        sp_raw = float(spearmanr(gold, raw)[0])

        # Left: scatter
        ax = axes[row, 0]
        sc = ax.scatter(gold, den, c=ps, cmap="RdYlBu_r", alpha=0.7, s=40,
                        edgecolor="#333", linewidth=0.3)
        ax.plot([1, 5], [1, 5], "--", color="#888", linewidth=1.0)
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0.8, 5.2)
        ax.set_xlabel("Gold score", fontsize=11)
        ax.set_ylabel("Predicted score", fontsize=11)
        ax.set_title(f"{label}\nr = {pr_den:.3f}, ρ = {sp_raw:.3f}", fontsize=11)
        ax.set_aspect("equal", adjustable="box")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("σ̂", fontsize=9)

        # Right: reliability
        ax = axes[row, 1]
        rc_raw = reliability_curve(raw, gold, n_bins=10, score_min=1.0, score_max=5.0)
        rc_den = reliability_curve(den, gold, n_bins=10, score_min=1.0, score_max=5.0)
        ece_r = expected_calibration_error(raw, gold, score_min=1.0, score_max=5.0)
        ece_d = expected_calibration_error(den, gold, score_min=1.0, score_max=5.0)

        ax.plot([1, 5], [1, 5], "--", color="#888", linewidth=1.0, label="Perfect")
        mask = rc_raw["counts"] > 0
        ax.plot(rc_raw["pred_means"][mask], rc_raw["label_means"][mask],
                "o-", linewidth=2, color="#d6604d", label=f"Raw (ECE={ece_r:.3f})")
        mask = rc_den["counts"] > 0
        ax.plot(rc_den["pred_means"][mask], rc_den["label_means"][mask],
                "s-", linewidth=2, color="#2166ac", label=f"Tweedie (ECE={ece_d:.3f})")
        ax.set_xlim(1, 5)
        ax.set_ylim(1, 5)
        ax.set_xlabel("Predicted score", fontsize=11)
        ax.set_ylabel("Mean gold per bin", fontsize=11)
        ax.set_title(f"Reliability — {label.split('(')[0].strip()}", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        "DiffuJudge-AV — Score Diffusion Judging: Synthetic vs. Real LingoQA\n"
        "3-judge Claude ensemble × 7 perturbation levels × 3 samples per level",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "synth_vs_lingoqa.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "synth_vs_lingoqa.svg", bbox_inches="tight")
    plt.close(fig)

    # ====== Score compression figure (LingoQA-specific) ======
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)

    # Panel A: box plot by gold tier
    ax = axes[0]
    tiers = [1, 2, 3, 4, 5]
    tier_data = []
    tier_labels = []
    for t in tiers:
        mask = (np.round(g_l) == t)
        if mask.sum() > 0:
            tier_data.append(r_l[mask])
            tier_labels.append(f"Gold≈{t}\n(n={mask.sum()})")
    bp = ax.boxplot(tier_data, labels=tier_labels, patch_artist=True,
                    boxprops=dict(facecolor="#92c5de", alpha=0.7),
                    medianprops=dict(color="#d6604d", linewidth=2))
    ax.plot([0.5, len(tiers)+0.5], [1, 5], "--", color="#888", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Predicted score (raw mean)", fontsize=11)
    ax.set_title("A. Score compression: Claude judges\ncluster predictions toward 2–3", fontsize=11)
    ax.set_ylim(0.5, 5.5)

    # Panel B: per-tier mean ± std
    ax = axes[1]
    means_gold = []
    means_raw = []
    stds_raw = []
    for t in tiers:
        mask = (np.round(g_l) == t)
        if mask.sum() > 0:
            means_gold.append(t)
            means_raw.append(float(r_l[mask].mean()))
            stds_raw.append(float(r_l[mask].std()))
    ax.errorbar(means_gold, means_raw, yerr=stds_raw, fmt="o-", color="#2166ac",
                linewidth=2, capsize=5, markersize=8, label="Claude prediction")
    ax.plot([1, 5], [1, 5], "--", color="#888", linewidth=1.0, label="Perfect calibration")
    ax.set_xlabel("Gold score tier", fontsize=11)
    ax.set_ylabel("Mean predicted score", fontsize=11)
    ax.set_title("B. Compression toward center\n(good rank order, compressed scale)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)

    fig.suptitle(
        "LingoQA: Claude judges show score compression on real AV-VQA\n"
        "Strong rank correlation (r=0.748) despite compressed output range",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lingoqa_compression.png", bbox_inches="tight")
    plt.close(fig)

    # ====== Summary metrics JSON ======
    summary = {
        "synthetic": {
            "n": len(g_s),
            "pearson": float(pearsonr(g_s, d_s)[0]),
            "spearman": float(spearmanr(g_s, d_s)[0]),
            "ece": float(expected_calibration_error(d_s, g_s, score_min=1, score_max=5)),
            "brier": float(brier_score(d_s, g_s)),
            "pred_range": [float(d_s.min()), float(d_s.max())],
            "gold_range": [float(g_s.min()), float(g_s.max())],
        },
        "lingoqa": {
            "n": len(g_l),
            "pearson": float(pearsonr(g_l, d_l)[0]),
            "spearman": float(spearmanr(g_l, d_l)[0]),
            "ece": float(expected_calibration_error(d_l, g_l, score_min=1, score_max=5)),
            "brier": float(brier_score(d_l, g_l)),
            "pred_range": [float(d_l.min()), float(d_l.max())],
            "gold_range": [float(g_l.min()), float(g_l.max())],
            "compression_ratio": float((d_l.max() - d_l.min()) / (g_l.max() - g_l.min())),
        },
    }
    with open(OUT_DIR / "synth_vs_lingoqa_metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\n[ok] figures → {OUT_DIR}/synth_vs_lingoqa.png, lingoqa_compression.png")


if __name__ == "__main__":
    main()
