"""Re-derive `summary.jsonl` from an existing `judge_outputs.jsonl`.

Useful when you've changed the Tweedie denoiser settings (bandwidth, precision
weighting) and want to recompute denoised estimates without re-calling any
APIs. Reads the cached judge outputs, re-runs the denoiser, and overwrites
`summary.jsonl` in place.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import orjson

from diffujudge.denoiser.tweedie import AnalyticalTweedieDenoiser


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--bandwidth", default="scott")
    p.add_argument("--precision-weight", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    bandwidth = float(args.bandwidth) if args.bandwidth not in {"scott", "silverman"} else args.bandwidth

    # Recover gold from the existing summary.
    existing_summary = list(open(run_dir / "summary.jsonl", "rb"))
    gold_lookup = {
        orjson.loads(l)["item_id"]: orjson.loads(l).get("gold")
        for l in existing_summary if l.strip()
    }

    # Bucket judge outputs by item, recording score + level.
    samples_by_item: dict[str, list[float]] = collections.defaultdict(list)
    levels_by_item: dict[str, list[int]] = collections.defaultdict(list)
    for line in open(run_dir / "judge_outputs.jsonl", "rb"):
        if not line.strip():
            continue
        r = orjson.loads(line)
        sid = r["sample_id"]
        # sample_id format: "<item_id>::t<level>::k<index>"
        try:
            level = int(sid.split("::t")[1].split("::")[0])
        except (IndexError, ValueError):
            level = 0
        samples_by_item[r["item_id"]].append(float(r["score"]))
        levels_by_item[r["item_id"]].append(level)

    den = AnalyticalTweedieDenoiser(
        bandwidth=bandwidth,
        precision_weight=args.precision_weight,
    )

    out_path = run_dir / "summary.jsonl"
    with open(out_path, "wb") as fh:
        n = 0
        for iid, scores in samples_by_item.items():
            arr = np.array(scores, dtype=np.float64)
            lvl = np.array(levels_by_item[iid], dtype=np.int64)
            est = den.denoise_item(iid, arr, lvl)
            row = {
                "item_id": iid,
                "point_estimate": est.point_estimate,
                "raw_mean": est.raw_mean,
                "posterior_var": est.posterior_var,
                "n_samples": est.n_samples,
                "level_means": est.level_means,
                "sigma_per_level": est.sigma_per_level,
                "gold": gold_lookup.get(iid),
            }
            fh.write(orjson.dumps(row, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY))
            fh.write(b"\n")
            n += 1
    print(f"[ok] re-denoised {n} items → {out_path}")
    print(f"     bandwidth={bandwidth}  precision_weight={args.precision_weight}")


if __name__ == "__main__":
    main()
