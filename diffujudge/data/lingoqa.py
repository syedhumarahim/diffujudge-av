"""LingoQA loader (Wayve, ECCV 2024).

The user is responsible for accepting Wayve's license and downloading the
distribution. Once `data_dir` points to an extracted LingoQA tree (with
`videos/` and `eval/` directories, or a HuggingFace cache), this loader emits
typed items ready for the perturbation cascade.

If `data_dir` does not exist, `iter_items` yields nothing — see
`SyntheticDataset` for an offline-friendly stand-in.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LingoQAItem:
    item_id: str
    question: str
    reference_answer: str
    candidate_answer: str | None
    video_id: str
    frames: list[str] = field(default_factory=list)
    lingo_judge_score: float | None = None
    behavior_label: str | None = None
    meta: dict = field(default_factory=dict)


class LingoQALoader:
    def __init__(self, data_dir: str | Path, split: str = "eval") -> None:
        self.data_dir = Path(data_dir)
        self.split = split

    def iter_items(self, n: int | None = None) -> Iterator[LingoQAItem]:
        manifest = self._discover_manifest()
        if manifest is None:
            return  # no real data → caller falls back to synthetic
        with open(manifest) as fh:
            records = json.load(fh) if manifest.suffix == ".json" else [
                json.loads(line) for line in fh if line.strip()
            ]
        for i, r in enumerate(records):
            if n is not None and i >= n:
                break
            video_id = r.get("video_id") or r.get("segment_id") or f"clip_{i}"
            item_id = r.get("item_id") or str(r.get("question_id", f"lingoqa_{i}"))
            ref = r.get("reference_answer") or r.get("answer") or r.get("reference", "")
            yield LingoQAItem(
                item_id=item_id,
                question=r["question"],
                reference_answer=ref,
                candidate_answer=r.get("candidate_answer"),
                video_id=str(video_id),
                frames=[str(p) for p in self._frames_for(video_id)],
                lingo_judge_score=r.get("lingo_judge") or r.get("gold_score"),
                behavior_label=r.get("behavior") or r.get("behavior_label"),
                meta={k: v for k, v in r.items()
                      if k not in {"question", "answer", "reference_answer", "candidate_answer"}},
            )

    def _discover_manifest(self) -> Path | None:
        candidates = [
            self.data_dir / "eval_corpus.jsonl",
            self.data_dir / f"{self.split}.json",
            self.data_dir / f"{self.split}.jsonl",
            self.data_dir / "eval" / "eval.json",
            self.data_dir / "eval" / "eval.jsonl",
        ]
        return next((c for c in candidates if c.exists()), None)

    def _frames_for(self, video_id: str) -> list[Path]:
        for d in [
            self.data_dir / "frames" / video_id,
            self.data_dir / "videos" / video_id,
            self.data_dir / video_id,
        ]:
            if d.exists():
                return sorted([p for p in d.iterdir() if p.suffix in {".jpg", ".png", ".jpeg"}])
        return []
