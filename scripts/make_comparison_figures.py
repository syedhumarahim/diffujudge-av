"""Generate publication-ready figures comparing Claude vs open-source model ensembles.

Produces:
  1. hero_comparison.png — 2×3 grid: scatter + reliability + perturbation stability
     for Claude (top) and open-source (bottom)
  2. tweedie_lift.png — bar chart showing Tweedie improvement per model family
  3. noise_profile.png — per-level variance comparison (Claude vs open-source)
  4. model_comparison_table.json — structured metrics for the article
"""
from __future__ import annotations

import collections
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

CLAUDE_DIR = Path("outputs/4f66efea222c4b39")
TOGETHER_DIR = Path("outputs/together_ensemble")
OUT_DIR = Path("docs/figures")

LEVEL_NAMES = {
    0: "Anchor", 1: "Option swap", 2: "Rubric\nparaphrase",
    3: "Criterion\nreorder", 4: "Score-ID\nswap", 5: "Temperature\nnoise",
    6: "Exemplar\nresample", 7: "Frame\nshuffle",
}

COLORS = {
    "claude": "#2166ac",
    "together": "#b2182b",
    "raw": "#d6604d",
    "denoised": "#2166ac",
    "perfect": "#888888",
}


def load_summary(run_dir: Path):
    rows = [orjson.loads(l) for l in open(run_dir / "summary.jsonl", "rb") if l.strip()]
    rows = [r for r in rows if r.get("gold") is not None]
    gold = np.array([r["gold"] for r in rows], dtype=np.float64)
    raw = np.array([r["raw_mean"] for r in rows], dtype=np.float64)
    den = np.array([r["point_estimate"] for r in rows], dtype=np.float64)
    post_std = np.array([r["posterior_var"] ** 0.5 for r in rows], dtype=np.float64)
    return rows, gold, raw, den, post_std


def load_outputs(run_dir: Path):
    records = []
    for line in open(run_dir / "judge_outputs.jsonl", "rb"):
        if line.strip():
            records.append(orjson.loads(line))
    return records


def parse_level(sample_id: str) -> int:
    try:
        return int(sample_id.split("::t")[1].split("::")[0])
    except (IndexError, ValueError):
        return 0


def compute_per_level_variance(records):
    by_item_level = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        level = parse_level(r["sample_id"])
        by_item_level[r["item_id"]][level].append(float(r["score"]))

    level_vars = {}
    for lvl in range(8):
        variances = []
        for iid, levels in by_item_level.items():
            if lvl in levels and len(levels[lvl]) >= 2:
                variances.append(float(np.var(levels[lvl], ddof=1)))
        if variances:
            level_vars[lvl] = float(np.mean(variances))
    return level_vars


def compute_per_level_deltas(records):
    by_item_level = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        level = parse_level(r["sample_id"])
        by_item_level[r["item_id"]][level].append(float(r["score"]))

    deltas = {}
    for lvl in range(8):
        abs_deltas = []
        for iid, levels in by_item_level.items():
            if 0 in levels and lvl in levels:
                d = abs(np.mean(levels[lvl]) - np.mean(levels[0]))
                abs_deltas.append(d)
        if abs_deltas:
            deltas[lvl] = float(np.mean(abs_deltas))
    return deltas


def per_judge_metrics(records, gold_map):
    by_judge = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        by_judge[r["judge"]][r["item_id"]].append(float(r["score"]))

    results = {}
    for judge, items in by_judge.items():
        ids_with_gold = [iid for iid in items if iid in gold_map]
        if not ids_with_gold:
            continue
        g = np.array([gold_map[iid] for iid in ids_with_gold])
        p = np.array([np.mean(items[iid]) for iid in ids_with_gold])
        results[judge] = {
            "pearson": float(pearsonr(g, p)[0]),
            "spearman": float(spearmanr(g, p)[0]),
            "ece": float(expected_calibration_error(p, g, score_min=1, score_max=5)),
            "brier": float(brier_score(p, g)),
            "mean_score": float(p.mean()),
            "std_score": float(p.std()),
            "pred_range": [float(p.min()), float(p.max())],
        }
    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    _, g_c, r_c, d_c, ps_c = load_summary(CLAUDE_DIR)
    _, g_t, r_t, d_t, ps_t = load_summary(TOGETHER_DIR)

    recs_c = load_outputs(CLAUDE_DIR)
    recs_t = load_outputs(TOGETHER_DIR)

    gold_c = {r["item_id"]: r["gold"] for r in [orjson.loads(l) for l in open(CLAUDE_DIR / "summary.jsonl", "rb") if l.strip()] if r.get("gold") is not None}
    gold_t = {r["item_id"]: r["gold"] for r in [orjson.loads(l) for l in open(TOGETHER_DIR / "summary.jsonl", "rb") if l.strip()] if r.get("gold") is not None}

    # ====== FIGURE 1: Hero 2×3 comparison ======
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=160)

    datasets = [
        ("Claude ensemble (3 judges)", g_c, r_c, d_c, ps_c, recs_c),
        ("Open-source ensemble (3 models)", g_t, r_t, d_t, ps_t, recs_t),
    ]

    for row, (label, gold, raw, den, ps, recs) in enumerate(datasets):
        pr_raw = float(pearsonr(gold, raw)[0])
        pr_den = float(pearsonr(gold, den)[0])
        sp_den = float(spearmanr(gold, den)[0])

        # Col 0: Scatter (denoised vs gold)
        ax = axes[row, 0]
        sc = ax.scatter(gold, den, c=ps, cmap="RdYlBu_r", alpha=0.7, s=40,
                        edgecolor="#333", linewidth=0.3)
        ax.plot([1, 5], [1, 5], "--", color=COLORS["perfect"], linewidth=1.0)
        ax.set_xlim(0.8, 5.2); ax.set_ylim(0.8, 5.2)
        ax.set_xlabel("Gold score", fontsize=11)
        ax.set_ylabel("Predicted (Tweedie)", fontsize=11)
        ax.set_title(f"{label}\nr = {pr_den:.3f}, ρ = {sp_den:.3f}", fontsize=11)
        ax.set_aspect("equal", adjustable="box")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("σ̂", fontsize=9)

        # Col 1: Reliability diagram
        ax = axes[row, 1]
        rc_raw = reliability_curve(raw, gold, n_bins=10, score_min=1.0, score_max=5.0)
        rc_den = reliability_curve(den, gold, n_bins=10, score_min=1.0, score_max=5.0)
        ece_r = expected_calibration_error(raw, gold, score_min=1.0, score_max=5.0)
        ece_d = expected_calibration_error(den, gold, score_min=1.0, score_max=5.0)

        ax.plot([1, 5], [1, 5], "--", color=COLORS["perfect"], linewidth=1.0, label="Perfect")
        mask = rc_raw["counts"] > 0
        ax.plot(rc_raw["pred_means"][mask], rc_raw["label_means"][mask],
                "o-", linewidth=2, color=COLORS["raw"], label=f"Raw (ECE={ece_r:.3f})")
        mask = rc_den["counts"] > 0
        ax.plot(rc_den["pred_means"][mask], rc_den["label_means"][mask],
                "s-", linewidth=2, color=COLORS["denoised"], label=f"Tweedie (ECE={ece_d:.3f})")
        ax.set_xlim(1, 5); ax.set_ylim(1, 5)
        ax.set_xlabel("Predicted score", fontsize=11)
        ax.set_ylabel("Mean gold per bin", fontsize=11)
        ax.set_title(f"Reliability — {label.split('(')[0].strip()}", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.set_aspect("equal", adjustable="box")

        # Col 2: Per-level stability
        ax = axes[row, 2]
        deltas = compute_per_level_deltas(recs)
        levels = sorted(deltas.keys())
        vals = [deltas[l] for l in levels]
        names = [LEVEL_NAMES.get(l, f"L{l}") for l in levels]
        colors = [COLORS["claude"] if row == 0 else COLORS["together"] for _ in levels]
        ax.bar(range(len(levels)), vals, color=colors, alpha=0.8, edgecolor="#333", linewidth=0.5)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(names, fontsize=8, rotation=0, ha="center")
        ax.set_ylabel("|Δ from anchor|", fontsize=11)
        ax.set_title(f"Perturbation sensitivity\n(mean |Δ| per level)", fontsize=11)
        ax.set_ylim(0, max(vals) * 1.3 if vals else 1)

    fig.suptitle(
        "DiffuJudge-AV — Claude vs. Open-Source Judge Ensembles on LingoQA\n"
        "Score Diffusion Judging: 7-level perturbation cascade × 3 samples/level",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hero_comparison.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "hero_comparison.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {OUT_DIR}/hero_comparison.png")

    # ====== FIGURE 2: Tweedie lift comparison ======
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)

    families = []
    # Claude
    pr_raw_c = float(pearsonr(g_c, r_c)[0])
    pr_den_c = float(pearsonr(g_c, d_c)[0])
    families.append(("Claude\nensemble", pr_raw_c, pr_den_c, ps_c.mean()))
    # Together
    pr_raw_t = float(pearsonr(g_t, r_t)[0])
    pr_den_t = float(pearsonr(g_t, d_t)[0])
    families.append(("Open-source\nensemble", pr_raw_t, pr_den_t, ps_t.mean()))

    # Per-model for Together
    pj_t = per_judge_metrics(recs_t, gold_t)
    for judge, m in sorted(pj_t.items()):
        short = judge.replace("Instruct-Turbo", "").replace("-Instruct", "").strip("-")
        if len(short) > 15:
            short = short[:14] + "…"
        families.append((short, m["pearson"], m["pearson"], 0))

    x = np.arange(len(families))
    width = 0.35
    raws = [f[1] for f in families]
    dens = [f[2] for f in families]

    bars_raw = ax.bar(x - width/2, raws, width, label="Raw mean",
                      color=COLORS["raw"], alpha=0.8, edgecolor="#333", linewidth=0.5)
    bars_den = ax.bar(x + width/2, dens, width, label="Tweedie denoised",
                      color=COLORS["denoised"], alpha=0.8, edgecolor="#333", linewidth=0.5)

    for i, (r, d) in enumerate(zip(raws, dens)):
        delta = d - r
        if abs(delta) > 0.001:
            ax.annotate(f"Δ={delta:+.3f}", (i + width/2, d),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, fontweight="bold",
                        color="#006400" if delta > 0 else "#8b0000")

    ax.set_xticks(x)
    ax.set_xticklabels([f[0] for f in families], fontsize=10)
    ax.set_ylabel("Pearson r with gold", fontsize=12)
    ax.set_title("Tweedie Denoising Lift: Claude vs. Open-Source Models\n"
                 "Tweedie helps noisy open-source models, confirms Claude's robustness",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.7, color="#888", linestyle=":", alpha=0.5, linewidth=0.8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "tweedie_lift.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {OUT_DIR}/tweedie_lift.png")

    # ====== FIGURE 3: Per-level noise profile comparison ======
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)

    vars_c = compute_per_level_variance(recs_c)
    vars_t = compute_per_level_variance(recs_t)
    all_levels = sorted(set(vars_c.keys()) | set(vars_t.keys()))

    # Panel A: variance bars
    ax = axes[0]
    x = np.arange(len(all_levels))
    width = 0.35
    v_c = [vars_c.get(l, 0) for l in all_levels]
    v_t = [vars_t.get(l, 0) for l in all_levels]
    ax.bar(x - width/2, v_c, width, label="Claude", color=COLORS["claude"], alpha=0.8)
    ax.bar(x + width/2, v_t, width, label="Open-source", color=COLORS["together"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LEVEL_NAMES.get(l, f"L{l}") for l in all_levels], fontsize=8)
    ax.set_ylabel("Mean within-level variance", fontsize=11)
    ax.set_title("A. Per-level score variance", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

    # Panel B: σ̂ distributions
    ax = axes[1]
    ax.hist(ps_c, bins=25, alpha=0.6, color=COLORS["claude"], label=f"Claude (μ={ps_c.mean():.3f})",
            density=True, edgecolor="white")
    ax.hist(ps_t, bins=25, alpha=0.6, color=COLORS["together"], label=f"Open-source (μ={ps_t.mean():.3f})",
            density=True, edgecolor="white")
    ax.set_xlabel("Posterior σ̂", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("B. Posterior uncertainty distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

    fig.suptitle(
        "Noise Profile: Claude judges are consistently low-variance;\n"
        "open-source models show heterogeneous noise that Tweedie exploits",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "noise_profile.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {OUT_DIR}/noise_profile.png")

    # ====== FIGURE 4: Score compression comparison ======
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)

    for idx, (label, gold, den, color) in enumerate([
        ("Claude ensemble", g_c, d_c, COLORS["claude"]),
        ("Open-source ensemble", g_t, d_t, COLORS["together"]),
    ]):
        ax = axes[idx]
        tiers = [1, 2, 3, 4, 5]
        means_gold = []
        means_pred = []
        stds_pred = []
        for t in tiers:
            mask = (np.round(gold) == t)
            if mask.sum() > 0:
                means_gold.append(t)
                means_pred.append(float(den[mask].mean()))
                stds_pred.append(float(den[mask].std()))
        ax.errorbar(means_gold, means_pred, yerr=stds_pred, fmt="o-", color=color,
                    linewidth=2, capsize=5, markersize=8, label="Predicted")
        ax.plot([1, 5], [1, 5], "--", color=COLORS["perfect"], linewidth=1.0, label="Perfect")
        comp = (den.max() - den.min()) / (gold.max() - gold.min()) if (gold.max() - gold.min()) > 0 else 0
        ax.set_xlabel("Gold score tier", fontsize=11)
        ax.set_ylabel("Mean predicted score", fontsize=11)
        ax.set_title(f"{label}\nCompression ratio: {comp:.2f}×", fontsize=11)
        ax.legend(fontsize=10)
        ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)

    fig.suptitle(
        "Score Compression: Claude vs. Open-Source\n"
        "Claude compresses scores to narrow range; open-source models may spread wider",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "score_compression_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {OUT_DIR}/score_compression_comparison.png")

    # ====== Summary metrics JSON ======
    pj_c = per_judge_metrics(recs_c, gold_c)
    pj_t = per_judge_metrics(recs_t, gold_t)

    summary = {
        "claude_ensemble": {
            "n": len(g_c),
            "n_records": len(recs_c),
            "pearson_raw": float(pearsonr(g_c, r_c)[0]),
            "pearson_denoised": float(pearsonr(g_c, d_c)[0]),
            "spearman_raw": float(spearmanr(g_c, r_c)[0]),
            "spearman_denoised": float(spearmanr(g_c, d_c)[0]),
            "ece_raw": float(expected_calibration_error(r_c, g_c, score_min=1, score_max=5)),
            "ece_denoised": float(expected_calibration_error(d_c, g_c, score_min=1, score_max=5)),
            "brier_raw": float(brier_score(r_c, g_c)),
            "brier_denoised": float(brier_score(d_c, g_c)),
            "mean_posterior_std": float(ps_c.mean()),
            "pred_range": [float(d_c.min()), float(d_c.max())],
            "compression_ratio": float((d_c.max() - d_c.min()) / (g_c.max() - g_c.min())),
            "per_judge": pj_c,
        },
        "together_ensemble": {
            "n": len(g_t),
            "n_records": len(recs_t),
            "pearson_raw": float(pearsonr(g_t, r_t)[0]),
            "pearson_denoised": float(pearsonr(g_t, d_t)[0]),
            "spearman_raw": float(spearmanr(g_t, r_t)[0]),
            "spearman_denoised": float(spearmanr(g_t, d_t)[0]),
            "ece_raw": float(expected_calibration_error(r_t, g_t, score_min=1, score_max=5)),
            "ece_denoised": float(expected_calibration_error(d_t, g_t, score_min=1, score_max=5)),
            "brier_raw": float(brier_score(r_t, g_t)),
            "brier_denoised": float(brier_score(d_t, g_t)),
            "mean_posterior_std": float(ps_t.mean()),
            "pred_range": [float(d_t.min()), float(d_t.max())],
            "compression_ratio": float((d_t.max() - d_t.min()) / (g_t.max() - g_t.min())),
            "per_judge": pj_t,
        },
        "per_level_variance": {
            "claude": compute_per_level_variance(recs_c),
            "together": compute_per_level_variance(recs_t),
        },
    }
    with open(OUT_DIR / "model_comparison_table.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[ok] metrics → {OUT_DIR / 'model_comparison_table.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
