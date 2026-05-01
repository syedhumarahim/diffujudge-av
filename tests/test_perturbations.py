from __future__ import annotations

import random

import pytest

from diffujudge.config import PerturbationConfig
from diffujudge.perturbations import PerturbationCascade
from diffujudge.perturbations.operators import (
    PromptView,
    criterion_reorder,
    frame_shuffle,
    option_swap,
    rubric_paraphrase,
    score_id_swap,
)


def _view() -> PromptView:
    return PromptView(
        item_id="test",
        question="Q?",
        rubric=["A", "B", "C", "D"],
        options=["yes", "no"],
        frames=["f0", "f1", "f2", "f3", "f4"],
        exemplars=[{"id": "e1"}, {"id": "e2"}, {"id": "e3"}],
        n_exemplars=2,
        score_id_format="arabic",
    )


def test_option_swap_reverses():
    out = option_swap(_view(), random.Random(0))
    assert out.options == ["no", "yes"]
    assert _view().options == ["yes", "no"], "operator must not mutate input"


def test_frame_shuffle_permutes_but_preserves_set():
    rng = random.Random(0)
    out = frame_shuffle(_view(), rng)
    assert sorted(out.frames) == ["f0", "f1", "f2", "f3", "f4"]


def test_score_id_swap_changes_format():
    rng = random.Random(0)
    out = score_id_swap(_view(), rng, target="roman")
    assert out.score_id_format == "roman"


def test_criterion_reorder_preserves_multiset():
    out = criterion_reorder(_view(), random.Random(0))
    assert sorted(out.rubric) == ["A", "B", "C", "D"]


def test_rubric_paraphrase_hedge_fallback():
    out = rubric_paraphrase(_view(), random.Random(0))
    assert all(p != orig for p, orig in zip(out.rubric, _view().rubric))


@pytest.mark.parametrize("samples", [1, 3, 5])
def test_cascade_emits_expected_count(samples: int):
    cfg = PerturbationConfig(samples_per_level=samples)
    casc = PerturbationCascade(cfg, base_seed=42)
    out = casc.apply(_view(), include_anchor=True)
    expected = 1 + len(casc.active_levels) * samples
    assert len(out) == expected


def test_cascade_seeds_are_deterministic():
    cfg = PerturbationConfig(samples_per_level=2)
    a = PerturbationCascade(cfg, base_seed=42).apply(_view())
    b = PerturbationCascade(cfg, base_seed=42).apply(_view())
    assert [s.sample_seed for s in a] == [s.sample_seed for s in b]
