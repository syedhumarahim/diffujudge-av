"""JSONL streaming I/O with resumable writes — required by the design's repro hygiene."""
from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import orjson


_DUMP_OPTS = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY


def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "wb") as fh:
        for r in records:
            fh.write(orjson.dumps(r, option=_DUMP_OPTS))
            fh.write(b"\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield orjson.loads(line)


class JsonlWriter:
    """Append-only writer that flushes per record so a crash leaves a valid prefix."""

    def __init__(self, path: Path | str, resume: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "ab" if resume and self.path.exists() else "wb"
        self._fh = open(self.path, mode)

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(orjson.dumps(record, option=_DUMP_OPTS))
        self._fh.write(b"\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
