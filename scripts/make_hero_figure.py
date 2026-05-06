"""Generate the README's hero figure: reliability diagram raw vs Tweedie-denoised.

Reads a finished pipeline run dir's `summary.jsonl` and produces:

    docs/figures/reliability.png       — main hero (raw vs denoised)
    docs/figures/scatter.png           — gold vs prediction
    docs/figures/per_level.png         — per-perturbation-level mean score
    docs/figures/headline.json         — numbers for the README

Usage:
    python scripts/make_hero_figure.py --run-dir outputs/<fingerprint>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import orjson

from diffujudge.metrics.agreement import cohen_kappa, krippendorff_alpha, pearson, spearman
from diffujudge.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", default="./docs/figures")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [orjson.loads(line) for line in open(run_dir / "summary.jsonl", "rb") if line.strip()]
    rows = [r for r in rows if r.get("gold") is not None]
    if not rows:
        raise SystemExit("No items with gold labels in summary.jsonl")

    gold = np.array([r["gold"] for r in rows], dtype=np.float64)
    raw = np.array([r["raw_mean"] for r in rows], dtype=np.float64)
    den = np.array([r["point_estimate"] for r in rows], dtype=np.float64)

    plt.style.use("seaborn-v0_8-whitegrid")

    # ---- Reliability diagram (THE hero figure) ----
    rc_raw = reliability_curve(raw, gold, n_bins=10, score_min=1.0, score_max=5.0)
    rc_den = reliability_curve(den, gold, n_bins=10, score_min=1.0, score_max=5.0)

    ece_raw = expected_calibration_error(raw, gold, score_min=1.0, score_max=5.0)
    ece_den = expected_calibration_error(den, gold, score_min=1.0, score_max=5.0)
    brier_raw = brier_score(raw, gold)
    brier_den = brier_score(den, gold)

    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=160)
    ax.plot([1, 5], [1, 5], color="#888", linestyle="--", linewidth=1.0, label="Perfect calibration")

    # Raw ensemble mean
    mask_raw = rc_raw["counts"] > 0
    ax.plot(
        rc_raw["pred_means"][mask_raw],
        rc_raw["label_means"][mask_raw],
        marker="o",
        linewidth=2.2,
        color="#d6604d",
        label=f"Raw ensemble mean — ECE={ece_raw:.3f}, Brier={brier_raw:.3f}",
    )

    # Tweedie-denoised
    mask_den = rc_den["counts"] > 0
    ax.plot(
        rc_den["pred_means"][mask_den],
        rc_den["label_means"][mask_den],
        marker="s",
        linewidth=2.2,
        color="#2166ac",
        label=f"Tweedie-denoised — ECE={ece_den:.3f}, Brier={brier_den:.3f}",
    )

    ax.set_xlim(1, 5)
    ax.set_ylim(1, 5)
    ax.set_xlabel("Predicted score", fontsize=12)
    ax.set_ylabel("Mean gold score per bin", fontsize=12)
    ax.set_title(
        "Reliability diagram — DiffuJudge-AV\n"
        f"Score Diffusion Judging on a 3-judge Anthropic ensemble  (n={len(gold)})",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_dir / "reliability.png", bbox_inches="tight")
    fig.savefig(out_dir / "reliability.svg", bbox_inches="tight")
    plt.close(fig)

    # ---- Scatter: gold vs prediction ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=160, sharey=True)
    for ax, vals, name, color in zip(
        axes,
        [raw, den],
        ["Raw ensemble mean", "Tweedie-denoised"],
        ["#d6604d", "#2166ac"],
    ):
        ax.scatter(gold, vals, color=color, alpha=0.6, s=36)
        ax.plot([1, 5], [1, 5], "--", color="#888", linewidth=0.9)
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0.8, 5.2)
        ax.set_xlabel("Gold score")
        ax.set_title(name, fontsize=11)
    axes[0].set_ylabel("Predicted score")
    fig.suptitle("Per-item gold vs. prediction", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Per-level mean score ----
    levels = sorted({int(k) for r in rows for k in r.get("level_means", {})})
    means = []
    for t in levels:
        vals = [r["level_means"].get(str(t)) or r["level_means"].get(t) for r in rows]
        vals = [v for v in vals if v is not None]
        means.append(float(np.mean(vals)) if vals else float("nan"))
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=160)
    bars = ax.bar([str(l) for l in levels], means, color="#4393c3", edgecolor="#08519c")
    ax.set_xlabel("Perturbation level t (0=anchor, 1=option-swap, 2=paraphrase, 3=criterion-reorder, 4=score-id, 5=temperature, 6=exemplar, 7=frame-shuffle)",
                  fontsize=8.5, wrap=True)
    ax.set_ylabel("Mean score across items")
    ax.set_title("Per-perturbation-level mean — forward diffusion process", fontsize=11)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "per_level.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Headline numbers JSON ----
    gold_q = np.clip(np.round(gold).astype(int), 1, 5)
    raw_q = np.clip(np.round(raw).astype(int), 1, 5)
    den_q = np.clip(np.round(den).astype(int), 1, 5)

    headline = {
        "n_items": len(gold),
        "kappa_raw": float(cohen_kappa(gold_q, raw_q, weights="quadratic")),
        "kappa_denoised": float(cohen_kappa(gold_q, den_q, weights="quadratic")),
        "alpha_raw": float(krippendorff_alpha(np.stack([gold_q.astype(float), raw_q.astype(float)]), level="ordinal")),
        "alpha_denoised": float(krippendorff_alpha(np.stack([gold_q.astype(float), den_q.astype(float)]), level="ordinal")),
        "pearson_raw": float(pearson(gold, raw)),
        "pearson_denoised": float(pearson(gold, den)),
        "spearman_raw": float(spearman(gold, raw)),
        "spearman_denoised": float(spearman(gold, den)),
        "ece_raw": float(ece_raw),
        "ece_denoised": float(ece_den),
        "brier_raw": float(brier_raw),
        "brier_denoised": float(brier_den),
    }
    with open(out_dir / "headline.json", "w") as fh:
        json.dump(headline, fh, indent=2)

    print(json.dumps(headline, indent=2))
    print(f"\n[ok] figures → {out_dir}/(reliability|scatter|per_level).png")


if __name__ == "__main__":
    main()
