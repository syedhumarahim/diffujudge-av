from __future__ import annotations

from diffujudge.judges import JudgeEnsemble, MockJudge, RubricRequest
from diffujudge.perturbations.operators import PromptView


def _request(item_id: str) -> RubricRequest:
    return RubricRequest(
        view=PromptView(item_id=item_id, question="Q?", rubric=["c1"]),
        score_scale=5,
        candidate_answer="some answer",
    )


def test_mock_judge_returns_in_range():
    j = MockJudge(gold_lookup={"x": 3.5})
    out = j.judge_one(_request("x"))
    assert 1.0 <= out.score <= 5.0
    assert out.judge_name == "mock-judge"


def test_ensemble_aggregates_per_sample():
    j1 = MockJudge(name="a", gold_lookup={"x": 3.5})
    j2 = MockJudge(name="b", gold_lookup={"x": 3.5}, family_bias=0.5)
    ens = JudgeEnsemble([j1, j2])

    req = _request("x")
    req.view.meta["sample_id"] = req.view.item_id
    flat, agg = ens.judge([req])
    assert len(flat) == 2
    assert len(agg) == 1
    assert {"a", "b"} == set(agg[0].per_judge)
