"""Streamlit dashboard — the design's hero figure delivery vehicle.

Run via `diffujudge dashboard --run-dir <path>` or directly:

    DIFFUJUDGE_RUN_DIR=outputs/abc123 streamlit run dashboard/streamlit_app.py

The app reads `summary.jsonl` and `eval_report.json` from the run dir and
renders:
  • the reliability diagram (raw vs. denoised) — the README's hero figure,
  • the conformal interval plot,
  • per-perturbation-level score distributions,
  • per-taxonomy-bucket κ.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import streamlit as st

from diffujudge.metrics.calibration import reliability_curve

st.set_page_config(page_title="DiffuJudge-AV", layout="wide")
st.title("DiffuJudge-AV — Calibration & Robustness Dashboard")

run_dir = Path(os.environ.get("DIFFUJUDGE_RUN_DIR", "."))
if not (run_dir / "summary.jsonl").exists():
    st.error(f"No summary.jsonl in {run_dir}. Pass DIFFUJUDGE_RUN_DIR=…")
    st.stop()

rows = [orjson.loads(line) for line in open(run_dir / "summary.jsonl", "rb") if line.strip()]
df = pd.DataFrame(rows)

with_gold = df[df["gold"].notna()].copy() if "gold" in df.columns else pd.DataFrame()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Items", len(df))
c2.metric("Items with gold", len(with_gold))
c3.metric("Mean posterior σ", f"{np.sqrt(df['posterior_var']).mean():.3f}")
c4.metric("Config fingerprint", run_dir.name)

st.markdown("---")

if not with_gold.empty:
    st.subheader("Reliability diagram — raw vs. Tweedie-denoised")
    rc_raw = reliability_curve(
        with_gold["raw_mean"].to_numpy(),
        with_gold["gold"].to_numpy(),
    )
    rc_den = reliability_curve(
        with_gold["point_estimate"].to_numpy(),
        with_gold["gold"].to_numpy(),
    )
    chart = pd.DataFrame(
        {
            "bin_center": rc_raw["centers"],
            "raw_pred": rc_raw["pred_means"],
            "raw_label": rc_raw["label_means"],
            "den_pred": rc_den["pred_means"],
            "den_label": rc_den["label_means"],
        }
    )
    st.line_chart(chart.set_index("bin_center")[["raw_label", "den_label"]])
    st.caption(
        "y = mean gold per bin; x = mean predicted per bin. "
        "Closer to y=x is better calibrated. Diffusion-denoised line should be tighter to diagonal."
    )

    st.subheader("Per-item: gold vs. denoised")
    st.scatter_chart(with_gold[["gold", "point_estimate"]].rename(columns={"point_estimate": "denoised"}))

    st.subheader("Posterior std-dev distribution")
    sigmas = np.sqrt(with_gold["posterior_var"]).to_numpy()
    st.bar_chart(pd.Series(sigmas).describe())

st.markdown("---")
st.subheader("Per-level mean score (forward diffusion process)")
if "level_means" in df.columns:
    levels = sorted({int(k) for d in df["level_means"].dropna() for k in d})
    means_per_level = []
    for t in levels:
        vals = [d.get(str(t)) or d.get(t) for d in df["level_means"].dropna()]
        vals = [v for v in vals if v is not None]
        means_per_level.append({"level": t, "mean": float(np.mean(vals)) if vals else float("nan")})
    st.bar_chart(pd.DataFrame(means_per_level).set_index("level"))

st.markdown("---")
report_path = run_dir / "eval_report.json"
if report_path.exists():
    st.subheader("Eval-of-eval report")
    st.json(json.loads(report_path.read_text()))
else:
    st.info("Run `python scripts/eval.py --run-dir <run_dir>` to generate eval_report.json")
