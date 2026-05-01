"""CODA-LM loader for the corner-case stress split (50 clips, per the design's §7).

CODA-LM is a corner-case driving dataset with LLM-graded QA. The loader is
structurally identical to LingoQALoader; we keep them separate so the eval
harness can label results with the originating dataset and so we can stress-
test out-of-distribution conformal coverage (the design's "expected
publishable failure mode").
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodaLMItem:
    item_id: str
    question: str
    reference_answer: str
    candidate_answer: str | None
    video_id: str
    frames: list[str] = field(default_factory=list)
    corner_case_type: str | None = None
    meta: dict = field(default_factory=dict)


class CodaLMLoader:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    def iter_items(self, n: int | None = None) -> Iterator[CodaLMItem]:
        manifest = self._discover_manifest()
        if manifest is None:
            return
        with open(manifest) as fh:
            records = json.load(fh) if manifest.suffix == ".json" else [
                json.loads(line) for line in fh if line.strip()
            ]
        for i, r in enumerate(records):
            if n is not None and i >= n:
                break
            video_id = r.get("video_id") or f"coda_{i}"
            yield CodaLMItem(
                item_id=str(r.get("question_id", f"coda_lm_{i}")),
                question=r["question"],
                reference_answer=r.get("answer", ""),
                candidate_answer=r.get("candidate_answer"),
                video_id=str(video_id),
                frames=[str(p) for p in self._frames_for(video_id)],
                corner_case_type=r.get("corner_case_type"),
                meta={k: v for k, v in r.items()},
            )

    def _discover_manifest(self) -> Path | None:
        for c in [
            self.data_dir / "coda_lm.json",
            self.data_dir / "stress.jsonl",
            self.data_dir / "test.json",
        ]:
            if c.exists():
                return c
        return None

    def _frames_for(self, video_id: str) -> list[Path]:
        for d in [self.data_dir / "frames" / video_id, self.data_dir / video_id]:
            if d.exists():
                return sorted([p for p in d.iterdir() if p.suffix in {".jpg", ".png"}])
        return []
