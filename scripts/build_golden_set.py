"""Compose the three-tier golden set described in §6 of the design.

Tier 1 — Lingo-Judge anchors (auto): reads from a Lingo-Judge prediction CSV.
Tier 2 — Supermajority synthetic: requires LiteLLMJudge (set API keys in .env).
Tier 3 — Manual (yours): a YAML / JSONL file with item_id → score that you
         hand-label. See docs/annotation_guide.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lingo-csv", default=None, help="Tier 1: csv with item_id,score,confidence")
    p.add_argument("--manual-jsonl", default=None, help="Tier 3: jsonl with {item_id, score}")
    p.add_argument("--out", default="./data/golden/golden_set.json")
    p.add_argument("--tier1-threshold", type=float, default=0.8)
    args = p.parse_args()

    from diffujudge.data.golden_set import build_three_tier_gold

    tier1: dict[str, tuple[float, float]] = {}
    if args.lingo_csv:
        import csv

        with open(args.lingo_csv) as fh:
            for row in csv.DictReader(fh):
                tier1[row["item_id"]] = (float(row["score"]), float(row.get("confidence", 1.0)))

    tier3: dict[str, float] = {}
    if args.manual_jsonl:
        with open(args.manual_jsonl) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    tier3[obj["item_id"]] = float(obj["score"])

    gold = build_three_tier_gold(
        tier1_lingo=tier1,
        tier3_manual=tier3,
        tier1_confidence_threshold=args.tier1_threshold,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(
            [{"item_id": g.item_id, "score": g.score, "provenance": g.provenance, "confidence": g.confidence}
             for g in gold.items],
            fh,
            indent=2,
        )
    print(f"[ok] wrote {len(gold.items)} golden items → {out_path}")


if __name__ == "__main__":
    main()
