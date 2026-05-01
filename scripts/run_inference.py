"""End-to-end run on synthetic OR LingoQA, optionally with API-fallback judges.

Examples:
    # offline-only: 200 synthetic items, three mock judges
    python scripts/run_inference.py --dataset synthetic --n 200

    # API-fallback ensemble (requires keys in .env)
    python scripts/run_inference.py --dataset synthetic --n 50 \
        --judges gpt-4o-mini,claude-3-5-haiku-20241022,gemini-1.5-flash

    # real LingoQA + open VLM ensemble (GPU required)
    python scripts/run_inference.py --dataset lingoqa --data-dir data/lingoqa \
        --judges Qwen/Qwen2.5-VL-7B-Instruct,OpenGVLab/InternVL2-8B,lmms-lab/llava-critic-7b \
        --backend vllm
"""
from __future__ import annotations

import argparse
from pathlib import Path

from diffujudge.config import DiffuJudgeConfig, JudgeConfig
from diffujudge.data.lingoqa import LingoQALoader
from diffujudge.data.synthetic import SyntheticDataset, SyntheticItem
from diffujudge.judges.base import BaseJudge
from diffujudge.judges.mock_judge import MockJudge
from diffujudge.pipeline import DiffuJudgePipeline


def _build_judges(names: list[str], backend: str, gold_lookup: dict[str, float]) -> list[BaseJudge]:
    judges: list[BaseJudge] = []
    for name in names:
        name = name.strip()
        if backend == "mock":
            judges.append(MockJudge(name=name, gold_lookup=gold_lookup))
        elif backend == "api":
            from diffujudge.judges.api_judge import LiteLLMJudge

            judges.append(LiteLLMJudge(name=name, model=name))
        elif backend == "vllm":
            from diffujudge.judges.vllm_judge import VLLMJudge

            judges.append(VLLMJudge(name=name.split("/")[-1], model=name))
        else:
            raise SystemExit(f"Unknown backend: {backend}")
    return judges


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["synthetic", "lingoqa"], default="synthetic")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judges", default="mock-a,mock-b,mock-c")
    p.add_argument("--backend", choices=["mock", "api", "vllm"], default="mock")
    p.add_argument("--output-dir", default="./outputs")
    args = p.parse_args()

    if args.dataset == "synthetic":
        ds = SyntheticDataset.build(n=args.n, seed=args.seed)
        gold = ds.gold_lookup()
        items = ds.items
    else:
        loader = LingoQALoader(args.data_dir)
        items = list(loader.iter_items(n=args.n))
        gold = {it.item_id: float(it.lingo_judge_score or 3.0) for it in items}

    if not items:
        raise SystemExit(
            "No items loaded — point --data-dir at a real LingoQA tree, or use --dataset synthetic."
        )

    cfg = DiffuJudgeConfig(
        seed=args.seed,
        output_dir=Path(args.output_dir),
        judges=[JudgeConfig(name=name.strip(), backend=args.backend, model=name.strip())
                for name in args.judges.split(",")],
    )
    judges = _build_judges(args.judges.split(","), args.backend, gold)
    pipe = DiffuJudgePipeline(cfg=cfg, judges=judges)

    # Adapt LingoQA → SyntheticItem-shaped (the pipeline only reads attributes).
    if args.dataset == "lingoqa":
        items = [
            SyntheticItem(
                item_id=it.item_id,
                question=it.question,
                reference_answer=it.reference_answer,
                candidate_answer=it.candidate_answer or "(no candidate)",
                behavior_label=it.behavior_label or "no_conflict",
                gold_score=gold[it.item_id],
                frames=it.frames,
                meta=it.meta,
            )
            for it in items
        ]

    res = pipe.run(items, gold=gold, output_dir=Path(args.output_dir))
    print(f"[ok] {len(res.estimates)} estimates")
    print(f"     raw judge outputs: {res.raw_judge_outputs_path}")
    print(f"     summary:           {res.metrics_path}")


if __name__ == "__main__":
    main()
