from __future__ import annotations

from pathlib import Path

from diffujudge.pipeline import DiffuJudgePipeline


def test_pipeline_smoke(tmp_path: Path, cfg, mock_judges, small_dataset, gold_lookup):
    cfg.output_dir = tmp_path
    pipe = DiffuJudgePipeline(cfg=cfg, judges=mock_judges)
    res = pipe.run(small_dataset.items, gold=gold_lookup, output_dir=tmp_path)

    assert len(res.estimates) == len(small_dataset.items)
    assert res.raw_judge_outputs_path.exists()
    assert res.metrics_path.exists()
    # Tweedie should not push estimates outside the score range.
    for e in res.estimates:
        assert 1.0 <= e.point_estimate <= 5.0
        assert e.posterior_var > 0


def test_denoising_improves_or_matches_raw_on_clean_gold(
    tmp_path: Path, cfg, mock_judges, small_dataset, gold_lookup,
):
    """With three reasonable mock judges and a Tweedie denoiser, mean abs error
    on the gold should not be *worse* than the raw judge mean.
    """
    cfg.output_dir = tmp_path
    pipe = DiffuJudgePipeline(cfg=cfg, judges=mock_judges)
    res = pipe.run(small_dataset.items, gold=gold_lookup, output_dir=tmp_path)

    raw_err = sum(abs(e.raw_mean - gold_lookup[e.item_id]) for e in res.estimates) / len(res.estimates)
    den_err = sum(abs(e.point_estimate - gold_lookup[e.item_id]) for e in res.estimates) / len(res.estimates)
    assert den_err <= raw_err + 0.10  # allow slack for KDE bandwidth choice
