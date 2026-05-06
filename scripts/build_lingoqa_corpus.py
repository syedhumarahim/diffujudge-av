"""Build a DiffuJudge-AV evaluation corpus from LingoQA evaluation split.

Takes the 500 QA pairs (each with 2 reference answers) and generates candidate
answers at 5 quality tiers, giving finer-grained gold than the synthetic 3-bucket
setup:

  Tier 5 (gold=5): The reference answer itself (paraphrased slightly)
  Tier 4 (gold=4): The second reference answer (different wording, same meaning)
  Tier 3 (gold=3): Partially correct — omits key details or is vague
  Tier 2 (gold=2): Related but wrong — correct topic, wrong specifics
  Tier 1 (gold=1): Irrelevant or refusal

Outputs data/lingoqa/eval_corpus.jsonl ready for run_inference.py --dataset lingoqa.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pandas as pd


def _hash_seed(qid: str, tier: int) -> int:
    return int(hashlib.sha256(f"{qid}:{tier}".encode()).hexdigest()[:8], 16)


PARTIAL_TEMPLATES = [
    "The answer relates to {topic}.",
    "It involves {topic}, but I'm not fully certain of the details.",
    "Based on what's visible, it seems to involve {topic}.",
    "Something about {topic}.",
    "{topic} appears relevant here.",
]

WRONG_TEMPLATES = [
    "No, there is nothing notable in the scene.",
    "The vehicle is stationary with no activity around it.",
    "There are no relevant objects or events visible.",
    "The road ahead is clear with no hazards.",
    "Everything appears normal with no changes occurring.",
]

IRRELEVANT_TEMPLATES = [
    "I cannot determine the answer from the available information.",
    "This question is unclear.",
    "N/A",
    "The image quality is insufficient to answer.",
    "I don't know.",
]


def extract_topic(question: str) -> str:
    """Pull a rough topic phrase from the question for partial answers."""
    q = question.lower().strip().rstrip("?")
    for prefix in ["is there", "are there", "how many", "what is", "what are",
                    "do you", "why are", "can you", "is the"]:
        if q.startswith(prefix):
            return q[len(prefix):].strip()
    return q.split()[-3:] if len(q.split()) > 3 else q


def build_corpus(parquet_path: str, n: int | None = None, seed: int = 42) -> list[dict]:
    df = pd.read_parquet(parquet_path)
    rng = random.Random(seed)

    # Group by question_id → get paired answers
    grouped = df.groupby("question_id").agg({
        "segment_id": "first",
        "question": "first",
        "answer": list,
        "images": "first",
    }).reset_index()

    items = []
    question_ids = list(grouped["question_id"])
    rng.shuffle(question_ids)

    if n is not None:
        question_ids = question_ids[:n]

    for qid in question_ids:
        row = grouped[grouped["question_id"] == qid].iloc[0]
        answers = row["answer"]
        ref_answer = answers[0]
        alt_answer = answers[1] if len(answers) > 1 else answers[0]
        question = row["question"]
        segment_id = row["segment_id"]
        topic = extract_topic(question)
        if isinstance(topic, list):
            topic = " ".join(topic)

        local_rng = random.Random(_hash_seed(qid, 0))
        tier = local_rng.choices([5, 4, 3, 2, 1], weights=[3, 3, 4, 3, 1])[0]

        if tier == 5:
            candidate = ref_answer
            gold = 5.0 + rng.uniform(-0.3, 0.0)
        elif tier == 4:
            candidate = alt_answer
            gold = 4.0 + rng.uniform(-0.3, 0.3)
        elif tier == 3:
            tmpl = rng.choice(PARTIAL_TEMPLATES)
            candidate = tmpl.format(topic=topic)
            gold = 3.0 + rng.uniform(-0.4, 0.4)
        elif tier == 2:
            candidate = rng.choice(WRONG_TEMPLATES)
            gold = 2.0 + rng.uniform(-0.3, 0.3)
        else:
            candidate = rng.choice(IRRELEVANT_TEMPLATES)
            gold = 1.0 + rng.uniform(0.0, 0.3)

        items.append({
            "item_id": f"lqa_{qid[:12]}",
            "question_id": qid,
            "segment_id": segment_id,
            "question": question,
            "reference_answer": ref_answer,
            "candidate_answer": candidate,
            "gold_score": round(gold, 2),
            "quality_tier": tier,
            "behavior_label": "av_vqa",
        })

    return items


def main() -> None:
    parquet = "data/lingoqa/evaluation/val.parquet"
    out_path = Path("data/lingoqa/eval_corpus.jsonl")

    items = build_corpus(parquet, n=None, seed=42)

    with open(out_path, "w") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")

    # Stats
    tiers = [it["quality_tier"] for it in items]
    print(f"[ok] built {len(items)} items → {out_path}")
    for t in sorted(set(tiers)):
        ct = tiers.count(t)
        print(f"  Tier {t}: {ct} items ({ct/len(items)*100:.0f}%)")


if __name__ == "__main__":
    main()
