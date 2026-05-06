"""Pydantic configuration for the DiffuJudge-AV pipeline.

All run-time knobs flow through these models so configs are typed, validated,
serializable to JSON/YAML, and reproducible from disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


ScoreScale = Literal[5, 7, 10]


class JudgeConfig(BaseModel):
    """A single judge in the ensemble."""

    name: str
    backend: Literal["vllm", "api", "anthropic", "anthropic_ensemble", "mock", "noisy_mock"] = "mock"
    model: str = "mock-judge"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    extra: dict = Field(default_factory=dict)


class PerturbationConfig(BaseModel):
    """Forward-diffusion perturbation cascade.

    Each entry corresponds to one noise level t with a known bias source from
    the 2024–25 literature. Order matches Section 4.3 of the design doc.
    """

    enable_option_swap: bool = True            # t=1, position bias
    enable_rubric_paraphrase: bool = True      # t=2, SPUQ-style
    enable_criterion_reorder: bool = True      # t=3, scoring bias
    enable_score_id_swap: bool = True          # t=4, Arabic↔Roman
    enable_temperature_noise: bool = True      # t=5
    enable_exemplar_resample: bool = True      # t=6, few-shot
    enable_frame_shuffle: bool = True          # t=7, video-only

    samples_per_level: int = 3                 # k in N×k tensor
    paraphrase_variants: int = 3
    temperature_grid: list[float] = Field(default_factory=lambda: [0.0, 0.3, 0.7])
    score_id_formats: list[str] = Field(default_factory=lambda: ["arabic", "roman", "alpha"])

    @property
    def num_levels(self) -> int:
        return sum(
            [
                self.enable_option_swap,
                self.enable_rubric_paraphrase,
                self.enable_criterion_reorder,
                self.enable_score_id_swap,
                self.enable_temperature_noise,
                self.enable_exemplar_resample,
                self.enable_frame_shuffle,
            ]
        )


class TweedieConfig(BaseModel):
    """Posterior-mean denoiser (Manor & Michaeli, ICLR 2024)."""

    mode: Literal["analytical", "learned", "both"] = "analytical"

    # Analytical KDE path.
    kde_bandwidth: float | Literal["scott", "silverman"] = "scott"
    sigma_per_level: list[float] | None = None  # auto-estimated if None
    score_min: float = 1.0
    score_max: float = 5.0
    score_grid_points: int = 256

    # Learned MLP path.
    mlp_hidden: int = 64
    mlp_depth: int = 2
    mlp_lr: float = 1e-3
    mlp_epochs: int = 50
    mlp_batch_size: int = 32


class ConformalConfig(BaseModel):
    """Ordinal conformal prediction wrapper (Sheng et al., EMNLP 2025)."""

    alpha: float = 0.10
    method: Literal["plain", "ordinal_boundary", "r2ccp", "chr"] = "ordinal_boundary"
    score_classes: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])

    @field_validator("alpha")
    @classmethod
    def _check_alpha(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError(f"alpha must be in (0, 1); got {v}")
        return v


class DataConfig(BaseModel):
    dataset: Literal["lingoqa", "coda_lm", "synthetic"] = "synthetic"
    data_dir: Path = Path("./data")
    n_items: int = 200
    n_frames_per_clip: int = 6
    frame_sampler: Literal["uniform", "scene_change"] = "uniform"
    cache_dir: Path = Path("./.cache")


class GoldenSetConfig(BaseModel):
    tier1_lingo_judge_threshold: float = 0.8
    tier2_supermajority_kappa: float = 0.8
    tier3_manual_target: int = 50
    calibration_split: float = 0.6  # vs. test


class EvalConfig(BaseModel):
    seeds: list[int] = Field(default_factory=lambda: [13, 17, 23, 31, 47])
    metrics: list[str] = Field(
        default_factory=lambda: [
            "cohen_kappa",
            "krippendorff_alpha",
            "pearson",
            "spearman",
            "kendall_tau",
            "ece",
            "mce",
            "brier",
            "conformal_coverage",
            "interval_width",
            "position_bias_delta",
            "verbosity_bias_delta",
            "scoring_id_bias_delta",
            "stochastic_stability",
        ]
    )


class DiffuJudgeConfig(BaseModel):
    """Top-level DiffuJudge-AV configuration."""

    name: str = "diffujudge-av-default"
    seed: int = 42
    score_scale: ScoreScale = 5
    output_dir: Path = Path("./outputs")

    judges: list[JudgeConfig] = Field(
        default_factory=lambda: [
            JudgeConfig(name="qwen2.5-vl-7b", backend="vllm", model="Qwen/Qwen2.5-VL-7B-Instruct"),
            JudgeConfig(name="internvl2-8b", backend="vllm", model="OpenGVLab/InternVL2-8B"),
            JudgeConfig(name="llava-critic-7b", backend="vllm", model="lmms-lab/llava-critic-7b"),
        ]
    )

    perturbations: PerturbationConfig = Field(default_factory=PerturbationConfig)
    tweedie: TweedieConfig = Field(default_factory=TweedieConfig)
    conformal: ConformalConfig = Field(default_factory=ConformalConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    golden_set: GoldenSetConfig = Field(default_factory=GoldenSetConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    def fingerprint(self) -> str:
        """Stable hash of the config — used to namespace outputs and caches."""
        import hashlib
        import orjson

        payload = orjson.dumps(self.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(payload).hexdigest()[:16]
