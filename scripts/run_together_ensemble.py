"""Run 3 open-source models on LingoQA via Together AI with parallel API calls.

Produces output compatible with the existing pipeline's judge_outputs.jsonl + summary.jsonl
format so downstream scripts (compute_bias_deltas, make_*_figures) work unchanged.
"""
from __future__ import annotations

import collections
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Load .env before any judge imports
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
except ImportError:
    pass

import litellm
from diffujudge.config import DiffuJudgeConfig
from diffujudge.data.lingoqa import LingoQALoader
from diffujudge.data.synthetic import SyntheticItem
from diffujudge.denoiser import AnalyticalTweedieDenoiser
from diffujudge.judges.api_judge import _build_prompt, parse_score_from_text
from diffujudge.judges.base import RubricRequest
from diffujudge.perturbations import PerturbationCascade, PromptView
from diffujudge.utils import JsonlWriter

MODELS = [
    "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "together_ai/deepseek-ai/DeepSeek-V3",
]

RUBRIC = [
    "Correctness — does the answer match the reference?",
    "Safety-criticality coverage — is the safety-critical event identified?",
    "Specificity — is the answer concrete (TTC, headway, lateral position)?",
    "Hallucination control — does the answer avoid unsupported claims?",
]

MAX_WORKERS = 30
MAX_TOKENS = 256


def build_view(item: SyntheticItem) -> PromptView:
    return PromptView(
        item_id=item.item_id,
        question=item.question,
        rubric=list(RUBRIC),
        score_id_format="arabic",
        options=[],
        frames=list(item.frames),
        exemplars=[],
        n_exemplars=0,
        temperature=0.0,
        meta={"behavior": item.behavior_label},
    )


def call_api(model: str, prompt: str, temperature: float) -> tuple[str, float, float]:
    t0 = time.perf_counter()
    try:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=MAX_TOKENS,
        )
        text = resp.choices[0].message.content or ""
        cost = float(getattr(resp, "_response_cost", 0.0) or 0.0)
        return text, cost, time.perf_counter() - t0
    except Exception as e:
        return f"(api-error) {type(e).__name__}: {e}", 0.0, time.perf_counter() - t0


def main():
    cfg = DiffuJudgeConfig(seed=42)
    cascade = PerturbationCascade(cfg.perturbations, base_seed=cfg.seed)

    loader = LingoQALoader("./data/lingoqa")
    raw_items = list(loader.iter_items(n=200))
    gold = {it.item_id: float(it.lingo_judge_score or 3.0) for it in raw_items}
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
        for it in raw_items
    ]

    print(f"Loaded {len(items)} items, generating perturbation views...")
    all_tasks = []
    for it in items:
        base_view = build_view(it)
        samples = cascade.apply(base_view, include_anchor=True)
        for s in samples:
            s.view.meta["sample_id"] = s.sample_id
            s.view.meta["perturb_level"] = s.level
            req = RubricRequest(
                view=s.view,
                score_scale=cfg.score_scale,
                reference_answer=it.reference_answer,
                candidate_answer=it.candidate_answer,
            )
            prompt = _build_prompt(req)
            for model in MODELS:
                all_tasks.append({
                    "item_id": s.item_id,
                    "sample_id": s.sample_id,
                    "level": s.level,
                    "model": model,
                    "judge_name": model.split("/")[-1],
                    "prompt": prompt,
                    "temperature": s.view.temperature or 0.0,
                    "fmt": s.view.score_id_format,
                })

    n_total = len(all_tasks)
    print(f"Total API calls: {n_total} ({len(items)} items × "
          f"{cascade.n_samples_per_item()} samples × {len(MODELS)} models)")

    out_dir = Path("outputs") / "together_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "judge_outputs.jsonl"

    done = 0
    errors = 0
    total_cost = 0.0
    lock = threading.Lock()
    writer = JsonlWriter(raw_path, resume=False)
    writer.__enter__()

    t_start = time.time()

    def process(task):
        nonlocal done, errors, total_cost
        text, cost, latency = call_api(task["model"], task["prompt"], task["temperature"])
        is_error = text.startswith("(api-error)")
        if is_error:
            score = 3.0
        else:
            score = parse_score_from_text(text, fmt=task["fmt"])
            score = max(1.0, min(5.0, score))

        record = {
            "item_id": task["item_id"],
            "sample_id": task["sample_id"],
            "judge": task["judge_name"],
            "score": score,
            "rationale": text[:500],
            "cost_usd": cost,
            "latency_s": latency,
            "meta": {"model": task["model"], "error": is_error},
        }

        with lock:
            writer.write(record)
            done += 1
            if is_error:
                errors += 1
            total_cost += cost
            if done % 200 == 0 or done == n_total:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_total - done) / rate if rate > 0 else 0
                print(f"  [{done:>6}/{n_total}] {rate:.1f} req/s | "
                      f"errors={errors} | cost=${total_cost:.3f} | "
                      f"ETA {eta/60:.1f}min")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(process, t) for t in all_tasks]
        for f in as_completed(futures):
            exc = f.exception()
            if exc:
                print(f"  [FATAL] {exc}")

    writer.__exit__(None, None, None)
    elapsed = time.time() - t_start
    print(f"\n[ok] {done} calls in {elapsed/60:.1f}min | "
          f"errors={errors} | cost=${total_cost:.4f}")
    print(f"     → {raw_path}")

    # Run denoiser and write summary
    print("\nRunning Tweedie denoiser...")
    records = []
    for line in open(raw_path, "rb"):
        if line.strip():
            records.append(json.loads(line))

    scores_by_item = collections.defaultdict(list)
    levels_by_item = collections.defaultdict(list)
    for r in records:
        scores_by_item[r["item_id"]].append(float(r["score"]))
        try:
            level = int(r["sample_id"].split("::t")[1].split("::")[0])
        except (IndexError, ValueError):
            level = 0
        levels_by_item[r["item_id"]].append(level)

    denoiser = AnalyticalTweedieDenoiser(score_min=1.0, score_max=5.0, bandwidth="scott")
    item_ids = list(scores_by_item.keys())
    scores_per = [np.array(scores_by_item[iid], dtype=np.float64) for iid in item_ids]
    levels_per = [np.array(levels_by_item[iid], dtype=np.int64) for iid in item_ids]
    estimates = denoiser.denoise_batch(item_ids, scores_per, levels_per)

    metrics_path = out_dir / "summary.jsonl"
    with JsonlWriter(metrics_path, resume=False) as w:
        for e in estimates:
            w.write({
                "item_id": e.item_id,
                "point_estimate": e.point_estimate,
                "raw_mean": e.raw_mean,
                "posterior_var": e.posterior_var,
                "n_samples": e.n_samples,
                "level_means": e.level_means,
                "sigma_per_level": e.sigma_per_level,
                "gold": gold.get(e.item_id),
            })

    # Quick metrics
    from scipy.stats import pearsonr, spearmanr
    items_with_gold = [iid for iid in item_ids if iid in gold]
    gold_arr = np.array([gold[iid] for iid in items_with_gold])
    raw_arr = np.array([np.mean(scores_by_item[iid]) for iid in items_with_gold])
    den_arr = np.array([next(e for e in estimates if e.item_id == iid).point_estimate
                        for iid in items_with_gold])
    post_std = np.array([next(e for e in estimates if e.item_id == iid).posterior_std
                         for iid in items_with_gold])

    print(f"\n{'='*60}")
    print("TOGETHER AI ENSEMBLE RESULTS (3 open-source models)")
    print(f"{'='*60}")
    print(f"  Items: {len(items_with_gold)}, Records: {len(records)}")
    print(f"  Pearson  raw={pearsonr(gold_arr, raw_arr)[0]:.4f}  "
          f"denoised={pearsonr(gold_arr, den_arr)[0]:.4f}  "
          f"Δ={pearsonr(gold_arr, den_arr)[0] - pearsonr(gold_arr, raw_arr)[0]:+.4f}")
    print(f"  Spearman raw={spearmanr(gold_arr, raw_arr)[0]:.4f}  "
          f"denoised={spearmanr(gold_arr, den_arr)[0]:.4f}  "
          f"Δ={spearmanr(gold_arr, den_arr)[0] - spearmanr(gold_arr, raw_arr)[0]:+.4f}")
    print(f"  Mean σ̂ = {post_std.mean():.4f}")
    print(f"  Pred range: [{den_arr.min():.2f}, {den_arr.max():.2f}]")
    print(f"  Gold range: [{gold_arr.min():.2f}, {gold_arr.max():.2f}]")

    # Save eval report
    report = {
        "models": [m.split("/")[-1] for m in MODELS],
        "n_items": len(items_with_gold),
        "n_records": len(records),
        "pearson_raw": float(pearsonr(gold_arr, raw_arr)[0]),
        "pearson_denoised": float(pearsonr(gold_arr, den_arr)[0]),
        "spearman_raw": float(spearmanr(gold_arr, raw_arr)[0]),
        "spearman_denoised": float(spearmanr(gold_arr, den_arr)[0]),
        "mean_posterior_std": float(post_std.mean()),
        "pred_range": [float(den_arr.min()), float(den_arr.max())],
    }
    with open(out_dir / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[ok] summary → {metrics_path}")
    print(f"[ok] report  → {out_dir / 'eval_report.json'}")


if __name__ == "__main__":
    main()
