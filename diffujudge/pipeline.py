"""End-to-end DiffuJudge-AV pipeline.

Pseudocode (matches the system diagram in the design's §5.1):

    items   = data.load(cfg.data)
    cascade = PerturbationCascade(cfg.perturbations)
    judges  = build_judges(cfg.judges)
    denoiser = AnalyticalTweedieDenoiser(...)
    conformal = OrdinalBoundaryConformal(...)

    for item in items:
        views = build_prompt_view(item)
        perturbed = cascade.apply(views)
        outputs = ensemble.judge(perturbed)
        # collect (score, level) per item
    estimates = denoiser.denoise_batch(...)
    conformal.fit(cal_predictions, cal_labels, cal_sigmas)
    intervals = conformal.predict(test_ids, test_predictions, test_sigmas)
    metrics = harness.run(estimates, gold)
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from diffujudge.config import DiffuJudgeConfig
from diffujudge.conformal import OrdinalBoundaryConformal
from diffujudge.data.synthetic import SyntheticItem
from diffujudge.denoiser import AnalyticalTweedieDenoiser, TweedieEstimate
from diffujudge.judges.base import BaseJudge, RubricRequest
from diffujudge.judges.ensemble import JudgeEnsemble
from diffujudge.perturbations import PerturbationCascade, PromptView
from diffujudge.utils import JsonlWriter, seed_everything


_DEFAULT_RUBRIC = [
    "Correctness — does the answer match the reference?",
    "Safety-criticality coverage — is the safety-critical event identified?",
    "Specificity — is the answer concrete (TTC, headway, lateral position)?",
    "Hallucination control — does the answer avoid unsupported claims?",
]


@dataclass
class PipelineRunResult:
    estimates: list[TweedieEstimate]
    intervals: list[Any]                              # ConformalInterval
    raw_judge_outputs_path: Path
    metrics_path: Path
    judge_score_matrix: dict[str, dict[int, list[float]]] = field(default_factory=dict)
    config_fingerprint: str = ""


class DiffuJudgePipeline:
    """One-shot orchestrator that wires the modules per the design's §5.1 diagram."""

    def __init__(
        self,
        cfg: DiffuJudgeConfig,
        judges: list[BaseJudge],
        rubric: list[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.ensemble = JudgeEnsemble(judges)
        self.cascade = PerturbationCascade(cfg.perturbations, base_seed=cfg.seed)
        self.denoiser = AnalyticalTweedieDenoiser(
            score_min=1.0,
            score_max=float(cfg.score_scale),
            bandwidth=cfg.tweedie.kde_bandwidth,
        )
        self.conformal = OrdinalBoundaryConformal(
            alpha=cfg.conformal.alpha,
            score_classes=tuple(cfg.conformal.score_classes),
            adaptive=True,
            boundary_snap=cfg.conformal.method == "ordinal_boundary",
        )
        self.rubric = rubric or _DEFAULT_RUBRIC

    def build_view(self, item: SyntheticItem) -> PromptView:
        """Convert a domain item into a judge-ready PromptView. Override per dataset."""
        return PromptView(
            item_id=item.item_id,
            question=item.question,
            rubric=list(self.rubric),
            score_id_format="arabic",
            options=[],
            frames=list(item.frames),
            exemplars=[],
            n_exemplars=0,
            temperature=0.0,
            meta={"behavior": item.behavior_label},
        )

    def run(
        self,
        items: Iterable[SyntheticItem],
        gold: dict[str, float],
        output_dir: Path | None = None,
    ) -> PipelineRunResult:
        seed_everything(self.cfg.seed)
        out = Path(output_dir or self.cfg.output_dir) / self.cfg.fingerprint()
        out.mkdir(parents=True, exist_ok=True)

        items = list(items)
        all_perturbed = []
        all_requests: list[RubricRequest] = []
        for it in items:
            base_view = self.build_view(it)
            samples = self.cascade.apply(base_view, include_anchor=True)
            for s in samples:
                s.view.meta["sample_id"] = s.sample_id
                s.view.meta["perturb_level"] = s.level
                s.view.meta[f"perturb_{s.level}"] = True
                all_perturbed.append(s)
                # Note: candidate_answer / reference_answer are NOT in PromptView;
                # they come through RubricRequest so the cascade only sees prompt-shape state.
                all_requests.append(
                    RubricRequest(
                        view=s.view,
                        score_scale=self.cfg.score_scale,
                        reference_answer=getattr(it, "reference_answer", None),
                        candidate_answer=getattr(it, "candidate_answer", None),
                    )
                )

        flat_outputs, _ = self.ensemble.judge(all_requests)

        raw_path = out / "judge_outputs.jsonl"
        with JsonlWriter(raw_path, resume=False) as w:
            for o in flat_outputs:
                w.write(
                    {
                        "item_id": o.item_id,
                        "sample_id": o.sample_id,
                        "judge": o.judge_name,
                        "score": o.score,
                        "rationale": o.rationale,
                        "cost_usd": o.cost_usd,
                        "latency_s": o.latency_s,
                        "meta": o.meta,
                    }
                )

        # Aggregate per item: pool across judges and across perturbation samples.
        scores_by_item: dict[str, list[float]] = defaultdict(list)
        levels_by_item: dict[str, list[int]] = defaultdict(list)
        sample_to_level = {p.sample_id: p.level for p in all_perturbed}
        for o in flat_outputs:
            scores_by_item[o.item_id].append(o.score)
            levels_by_item[o.item_id].append(sample_to_level.get(o.sample_id, 0))

        item_ids = list(scores_by_item)
        scores_per = [np.asarray(scores_by_item[iid], dtype=np.float64) for iid in item_ids]
        levels_per = [np.asarray(levels_by_item[iid], dtype=np.int64) for iid in item_ids]

        estimates = self.denoiser.denoise_batch(item_ids, scores_per, levels_per)
        est_by_id = {e.item_id: e for e in estimates}

        # Conformal calibration: random 60/40 split of items with a gold label.
        labeled_ids = [iid for iid in item_ids if iid in gold]
        rng = np.random.default_rng(self.cfg.seed)
        rng.shuffle(labeled_ids)
        cal_n = max(int(round(0.6 * len(labeled_ids))), 10) if labeled_ids else 0

        intervals = []
        if cal_n >= 10:
            cal_ids = labeled_ids[:cal_n]
            test_ids = labeled_ids[cal_n:]

            cal_pred = np.array([est_by_id[i].point_estimate for i in cal_ids])
            cal_lab = np.array([gold[i] for i in cal_ids])
            cal_sig = np.array([est_by_id[i].posterior_std for i in cal_ids])

            self.conformal.fit(cal_pred, cal_lab, cal_sigmas=cal_sig)

            test_pred = np.array([est_by_id[i].point_estimate for i in test_ids])
            test_sig = np.array([est_by_id[i].posterior_std for i in test_ids])
            intervals = self.conformal.predict(test_ids, test_pred, sigmas=test_sig)

        # Persist metrics summary.
        metrics_path = out / "summary.jsonl"
        with JsonlWriter(metrics_path, resume=False) as w:
            for e in estimates:
                w.write(
                    {
                        "item_id": e.item_id,
                        "point_estimate": e.point_estimate,
                        "raw_mean": e.raw_mean,
                        "posterior_var": e.posterior_var,
                        "n_samples": e.n_samples,
                        "level_means": e.level_means,
                        "sigma_per_level": e.sigma_per_level,
                        "gold": gold.get(e.item_id),
                    }
                )

        return PipelineRunResult(
            estimates=estimates,
            intervals=intervals,
            raw_judge_outputs_path=raw_path,
            metrics_path=metrics_path,
            config_fingerprint=self.cfg.fingerprint(),
        )
