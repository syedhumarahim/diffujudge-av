from __future__ import annotations

import numpy as np
import pytest

from diffujudge.config import DiffuJudgeConfig, JudgeConfig, PerturbationConfig
from diffujudge.data.synthetic import SyntheticDataset
from diffujudge.judges.mock_judge import MockJudge


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_dataset() -> SyntheticDataset:
    return SyntheticDataset.build(n=20, seed=7)


@pytest.fixture
def gold_lookup(small_dataset) -> dict[str, float]:
    return small_dataset.gold_lookup()


@pytest.fixture
def mock_judges(gold_lookup):
    return [
        MockJudge(name="mock-a", gold_lookup=gold_lookup),
        MockJudge(name="mock-b", gold_lookup=gold_lookup, family_bias=0.05),
        MockJudge(name="mock-c", gold_lookup=gold_lookup, verbosity_slope=0.02),
    ]


@pytest.fixture
def cfg() -> DiffuJudgeConfig:
    return DiffuJudgeConfig(
        seed=7,
        score_scale=5,
        judges=[JudgeConfig(name=n, backend="mock") for n in ("mock-a", "mock-b", "mock-c")],
        perturbations=PerturbationConfig(samples_per_level=2),
    )
