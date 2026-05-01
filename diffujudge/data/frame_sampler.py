"""Frame samplers for video judges.

`uniform_sample` is the design's recommended default for Qwen2.5-VL (5–8
frames per clip). `scene_change_sample` is the optional event-aware variant —
it picks frames where consecutive-frame absolute difference spikes, which
biases the sample toward salient transitions (cut-ins, brake-light onsets).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


def uniform_sample(n_total_frames: int, k: int) -> list[int]:
    if k <= 0:
        return []
    if n_total_frames <= k:
        return list(range(n_total_frames))
    return [int(round(i * (n_total_frames - 1) / max(k - 1, 1))) for i in range(k)]


def scene_change_sample(frames_paths: list[Path], k: int) -> list[int]:
    """Pick `k` frames at the highest absolute-pixel-difference transitions.

    Falls back to uniform if PIL is not available.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return uniform_sample(len(frames_paths), k)

    if len(frames_paths) <= k:
        return list(range(len(frames_paths)))

    arrs: list[np.ndarray] = []
    for p in frames_paths:
        im = Image.open(p).convert("L").resize((64, 64))
        arrs.append(np.asarray(im, dtype=np.float32))
    diffs = np.array([
        float(np.abs(arrs[i] - arrs[i - 1]).mean()) for i in range(1, len(arrs))
    ])
    diffs = np.concatenate(([0.0], diffs))
    # Greedy top-k with a min-spacing constraint to avoid clustering.
    order = np.argsort(-diffs)
    selected: list[int] = []
    spacing = max(len(frames_paths) // (k * 2), 1)
    for i in order:
        if all(abs(int(i) - s) >= spacing for s in selected):
            selected.append(int(i))
        if len(selected) == k:
            break
    return sorted(selected)


def sample_frames(
    frames: list[Path] | list[str],
    k: int,
    method: Literal["uniform", "scene_change"] = "uniform",
) -> list[Path]:
    paths = [Path(p) for p in frames]
    if method == "scene_change":
        idx = scene_change_sample(paths, k)
    else:
        idx = uniform_sample(len(paths), k)
    return [paths[i] for i in idx]
