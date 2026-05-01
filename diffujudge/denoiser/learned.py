"""Small learned MLP denoiser as ablation against the analytical Tweedie path.

Predicts the latent score ŝ_0 from a per-item feature vector summarizing the
N×k perturbed-score distribution. Trained on a held-out calibration slice of
the golden set (~120 items per the design's 60/40 split).

Architecture is intentionally tiny (2 hidden layers, 64 units) — this is the
ablation knob, not the main contribution. Borrowed in spirit from SiDyP
(Cao et al., KDD 2025, arXiv 2505.19675), which uses a simplex denoiser for
LLM-generated noisy labels.

Falls back gracefully if torch is not installed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


@dataclass
class LearnedTweedieResult:
    item_id: str
    point_estimate: float
    posterior_var: float
    raw_mean: float
    n_samples: int


def _featurize(
    scores: np.ndarray,
    levels: np.ndarray,
    n_levels: int,
) -> np.ndarray:
    """Compact per-item feature: [mean, std, min, max, q25, q75] + per-level mean.

    Matches the design's spec: f_θ(s̃, t, x_embed). We use the empirical score
    distribution as `s̃`-summary; an embed-based variant can be added by passing
    `extra_features` to `.fit` / `.predict`.
    """
    feats: list[float] = [
        float(scores.mean()),
        float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
        float(scores.min()),
        float(scores.max()),
        float(np.percentile(scores, 25)),
        float(np.percentile(scores, 75)),
    ]
    for t in range(n_levels + 1):
        bucket = scores[levels == t]
        feats.append(float(bucket.mean()) if bucket.size else 0.0)
    return np.asarray(feats, dtype=np.float32)


if _HAS_TORCH:

    class _MLP(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 64, depth: int = 2) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            d = in_dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.GELU()]
                d = hidden
            layers.append(nn.Linear(d, 2))  # (mean, log_var)
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class LearnedTweedieDenoiser:
        def __init__(
            self,
            n_levels: int = 7,
            hidden: int = 64,
            depth: int = 2,
            lr: float = 1e-3,
            epochs: int = 50,
            batch_size: int = 32,
            score_min: float = 1.0,
            score_max: float = 5.0,
            device: str | None = None,
        ) -> None:
            self.n_levels = n_levels
            self.hidden = hidden
            self.depth = depth
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.score_min = score_min
            self.score_max = score_max
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model: _MLP | None = None
            self._feat_dim: int | None = None

        def _features_for_items(
            self,
            scores_per_item: list[np.ndarray],
            levels_per_item: list[np.ndarray],
        ) -> np.ndarray:
            return np.stack(
                [_featurize(s, l, self.n_levels) for s, l in zip(scores_per_item, levels_per_item, strict=True)]
            )

        def fit(
            self,
            scores_per_item: list[np.ndarray],
            levels_per_item: list[np.ndarray],
            gold_scores: np.ndarray,
        ) -> "LearnedTweedieDenoiser":
            X = self._features_for_items(scores_per_item, levels_per_item)
            y = np.asarray(gold_scores, dtype=np.float32).ravel()
            self._feat_dim = X.shape[1]

            X_t = torch.from_numpy(X).to(self.device)
            y_t = torch.from_numpy(y).to(self.device)

            self._model = _MLP(self._feat_dim, hidden=self.hidden, depth=self.depth).to(self.device)
            opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)

            ds = TensorDataset(X_t, y_t)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

            self._model.train()
            for _ in range(self.epochs):
                for xb, yb in loader:
                    pred = self._model(xb)
                    mu, log_var = pred[:, 0], pred[:, 1].clamp(-6, 4)
                    var = log_var.exp()
                    # Heteroscedastic Gaussian NLL.
                    loss = (0.5 * ((yb - mu) ** 2) / var + 0.5 * log_var).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            return self

        def predict(
            self,
            item_ids: list[str],
            scores_per_item: list[np.ndarray],
            levels_per_item: list[np.ndarray],
        ) -> list[LearnedTweedieResult]:
            if self._model is None:
                raise RuntimeError("Call .fit(...) before .predict(...)")
            X = self._features_for_items(scores_per_item, levels_per_item)
            X_t = torch.from_numpy(X).to(self.device)
            self._model.eval()
            with torch.no_grad():
                pred = self._model(X_t).cpu().numpy()
            mu = np.clip(pred[:, 0], self.score_min, self.score_max)
            var = np.exp(np.clip(pred[:, 1], -6, 4))
            return [
                LearnedTweedieResult(
                    item_id=iid,
                    point_estimate=float(mu[i]),
                    posterior_var=float(var[i]),
                    raw_mean=float(scores_per_item[i].mean()),
                    n_samples=int(scores_per_item[i].size),
                )
                for i, iid in enumerate(item_ids)
            ]
