"""Typer CLI exposed as the `diffujudge` entry point.

  diffujudge run-synthetic --n 200 --seed 42 --judges mock,mock,mock
  diffujudge eval --run-dir outputs/<fingerprint>
  diffujudge dashboard --run-dir outputs/<fingerprint>
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="DiffuJudge-AV CLI")


@app.command("run-synthetic")
def run_synthetic(
    n: int = typer.Option(200, help="Number of synthetic items"),
    seed: int = typer.Option(42),
    judges: str = typer.Option("mock,mock-2,mock-3", help="Comma-separated judge names"),
    output_dir: Path = typer.Option(Path("./outputs")),
) -> None:
    """End-to-end run on the offline synthetic AV-flavored corpus."""
    from diffujudge.config import DiffuJudgeConfig, JudgeConfig
    from diffujudge.data.synthetic import SyntheticDataset
    from diffujudge.judges.mock_judge import MockJudge
    from diffujudge.pipeline import DiffuJudgePipeline

    ds = SyntheticDataset.build(n=n, seed=seed)
    gold = ds.gold_lookup()

    cfg = DiffuJudgeConfig(
        seed=seed,
        output_dir=output_dir,
        judges=[JudgeConfig(name=name.strip(), backend="mock") for name in judges.split(",")],
    )
    judge_objs = [
        MockJudge(name=jc.name, gold_lookup=gold, family_bias=0.0, verbosity_slope=0.05)
        for jc in cfg.judges
    ]
    pipe = DiffuJudgePipeline(cfg=cfg, judges=judge_objs)
    res = pipe.run(ds.items, gold=gold, output_dir=output_dir)

    typer.echo(f"[ok] wrote raw outputs to    {res.raw_judge_outputs_path}")
    typer.echo(f"[ok] wrote summary to        {res.metrics_path}")
    typer.echo(f"[ok] {len(res.estimates)} items denoised; {len(res.intervals)} conformal intervals")


@app.command("eval")
def eval_run(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False),
) -> None:
    """Run the eval-of-eval harness on a finished pipeline output dir."""
    import numpy as np
    import orjson

    from diffujudge.eval import EvalOfEvalHarness

    summary_path = run_dir / "summary.jsonl"
    rows = [orjson.loads(line) for line in open(summary_path, "rb") if line.strip()]
    rows = [r for r in rows if r.get("gold") is not None]
    item_ids = [r["item_id"] for r in rows]
    den = np.array([r["point_estimate"] for r in rows])
    raw = np.array([r["raw_mean"] for r in rows])
    gold = {r["item_id"]: float(r["gold"]) for r in rows}

    rep = EvalOfEvalHarness().run(item_ids, den, raw, gold)
    out = run_dir / "eval_report.json"
    rep.save(out)
    typer.echo(f"[ok] wrote {out}")
    typer.echo(
        f"  κ={rep.cohen_kappa:.3f}  α={rep.krippendorff_alpha:.3f}  "
        f"Pearson={rep.pearson:.3f}  ECE: {rep.ece_baseline:.3f} → {rep.ece_denoised:.3f}"
    )


@app.command("dashboard")
def dashboard(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False),
) -> None:
    """Launch the Streamlit dashboard against a finished pipeline run."""
    import os
    import subprocess

    app_path = Path(__file__).parent.parent / "dashboard" / "streamlit_app.py"
    env = os.environ.copy()
    env["DIFFUJUDGE_RUN_DIR"] = str(run_dir)
    subprocess.run(["streamlit", "run", str(app_path)], env=env, check=False)


if __name__ == "__main__":
    app()
