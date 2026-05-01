"""Pre-cache rubric paraphrases offline so the cascade runs deterministic.

Per the design's §5.1: "Generate paraphrases offline once; freeze them." This
script calls a small LLM (default GPT-4o-mini) to produce K paraphrases per
rubric criterion and writes them to a JSON cache that
`diffujudge.perturbations.operators.register_paraphrases` reads at startup.

If no API key is set, falls back to a deterministic rule-based paraphraser
(swap a few connectives) — adequate for tests, but the "real" run should use
an LLM for variety.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


_PROMPT = (
    "Paraphrase the following evaluation rubric criterion in {k} different ways. "
    "Keep the meaning identical; vary diction, sentence structure, and length. "
    "Return a JSON list of strings.\n\n"
    "Criterion: {criterion}\n"
)


def _llm_paraphrase(criterion: str, k: int) -> list[str]:
    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_paraphrase(criterion, k)
    try:
        import litellm
    except ImportError:
        return _fallback_paraphrase(criterion, k)
    resp = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": _PROMPT.format(k=k, criterion=criterion)}],
        temperature=0.7,
        max_tokens=512,
    )
    text = resp.choices[0].message.content
    try:
        arr = json.loads(text[text.find("[") : text.rfind("]") + 1])
        return [str(x) for x in arr][:k]
    except (ValueError, json.JSONDecodeError):
        return _fallback_paraphrase(criterion, k)


def _fallback_paraphrase(criterion: str, k: int) -> list[str]:
    hedges = ["Specifically: ", "In other words, ", "That is, ", "Equivalently, "]
    out = []
    for i in range(k):
        out.append(hedges[i % len(hedges)] + criterion[0].lower() + criterion[1:])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rubric-json", required=True, help="JSON {rubric_id: [criterion, ...]}")
    p.add_argument("--out", default="./.cache/paraphrases.json")
    p.add_argument("-k", "--variants", type=int, default=3)
    args = p.parse_args()

    rubrics = json.loads(Path(args.rubric_json).read_text())
    cache: dict[str, list[str]] = {}
    for rubric_id, criteria in rubrics.items():
        for i, c in enumerate(criteria):
            cache[f"{rubric_id}:crit{i}"] = _llm_paraphrase(c, k=args.variants)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, indent=2))
    print(f"[ok] wrote {len(cache)} entries → {out}")


if __name__ == "__main__":
    main()
