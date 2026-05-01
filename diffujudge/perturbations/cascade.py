"""Forward-diffusion cascade: emit N×k perturbed views per item.

Concretely: for every active perturbation level t, draw `samples_per_level`
fresh seeds and apply that level's operator to the base PromptView. The output
is a flat list of `PerturbedSample` records, each tagged with its noise level
and a deterministic per-sample seed for reproducibility.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from diffujudge.config import PerturbationConfig
from diffujudge.perturbations.operators import (
    OPERATOR_BY_LEVEL,
    PromptView,
    score_id_swap,
    temperature_noise,
)


@dataclass
class PerturbedSample:
    item_id: str
    sample_id: str
    level: int                      # noise level t; 0 for the un-perturbed anchor
    level_name: str
    sample_seed: int
    view: PromptView
    parent_meta: dict[str, Any] = field(default_factory=dict)


_LEVEL_NAMES: dict[int, str] = {
    0: "anchor",
    1: "option_swap",
    2: "rubric_paraphrase",
    3: "criterion_reorder",
    4: "score_id_swap",
    5: "temperature_noise",
    6: "exemplar_resample",
    7: "frame_shuffle",
}


class PerturbationCascade:
    """Apply a configured subset of the 7-level cascade to a PromptView."""

    def __init__(self, cfg: PerturbationConfig, base_seed: int = 42) -> None:
        self.cfg = cfg
        self.base_seed = base_seed
        self._active_levels = self._resolve_active_levels()

    def _resolve_active_levels(self) -> list[int]:
        flags = [
            (1, self.cfg.enable_option_swap),
            (2, self.cfg.enable_rubric_paraphrase),
            (3, self.cfg.enable_criterion_reorder),
            (4, self.cfg.enable_score_id_swap),
            (5, self.cfg.enable_temperature_noise),
            (6, self.cfg.enable_exemplar_resample),
            (7, self.cfg.enable_frame_shuffle),
        ]
        return [t for t, on in flags if on]

    @property
    def active_levels(self) -> list[int]:
        return list(self._active_levels)

    def n_samples_per_item(self, include_anchor: bool = True) -> int:
        return (1 if include_anchor else 0) + len(self._active_levels) * self.cfg.samples_per_level

    def apply(self, base: PromptView, include_anchor: bool = True) -> list[PerturbedSample]:
        out: list[PerturbedSample] = []

        if include_anchor:
            anchor_seed = self._derive_seed(base.item_id, level=0, k=0)
            out.append(
                PerturbedSample(
                    item_id=base.item_id,
                    sample_id=f"{base.item_id}::t0::k0",
                    level=0,
                    level_name=_LEVEL_NAMES[0],
                    sample_seed=anchor_seed,
                    view=base.copy(),
                )
            )

        for t in self._active_levels:
            op = OPERATOR_BY_LEVEL[t]
            for k in range(self.cfg.samples_per_level):
                seed = self._derive_seed(base.item_id, level=t, k=k)
                rng = random.Random(seed)

                if t == 4:
                    target = self.cfg.score_id_formats[k % len(self.cfg.score_id_formats)]
                    view = score_id_swap(base, rng, target=target)
                elif t == 5:
                    view = temperature_noise(base, rng, grid=self.cfg.temperature_grid)
                else:
                    view = op(base, rng)

                out.append(
                    PerturbedSample(
                        item_id=base.item_id,
                        sample_id=f"{base.item_id}::t{t}::k{k}",
                        level=t,
                        level_name=_LEVEL_NAMES[t],
                        sample_seed=seed,
                        view=view,
                        parent_meta=dict(view.meta),
                    )
                )
        return out

    def _derive_seed(self, item_id: str, level: int, k: int) -> int:
        # Hash-mix the (run_seed, item_id, level, k) tuple so reruns are exactly reproducible.
        h = hash((self.base_seed, item_id, level, k)) & 0x7FFFFFFF
        return int(h)
