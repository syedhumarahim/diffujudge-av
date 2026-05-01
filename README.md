# DiffuJudge-AV

**Calibrated, diffusion-style LLM/VLM-as-a-Judge — with a flagship VLM evaluator for safety-critical driving video.**

> Treat the LLM judge's score as a noisy observation along a *known* perturbation diffusion process whose noise levels correspond one-to-one to the canonical bias sources documented in the 2024–25 LLM-as-a-Judge literature. Recover the latent rubric score with a single-step Tweedie posterior-mean denoiser. Wrap the result in an ordinal-boundary-adjusted conformal prediction interval. Apply the whole stack to a 200-clip LingoQA subset plus a 50-clip CODA-LM corner-case stress split, scored against a three-tier golden set.

The framework is general-purpose; the AV video evaluator is one flagship application.

---

## Why this matters

Existing VLM-AV benchmarks evaluate the **AV system**. This repo evaluates the
**evaluator** — and gives calibrated confidence on its own judgments.

It speaks the dialect of the field: NHTSA pre-crash typology, ASAM
OpenSCENARIO, Cohen's κ / Krippendorff's α, ECE / Brier, conformal coverage,
position / verbosity / scoring-ID bias deltas, and a 12-category behavior
taxonomy aligned with CARLA Leaderboard scenarios.

## Headline contributions

1. **Score Diffusion Judging (SDJ).** A 7-level forward perturbation cascade
   that operationalizes the canonical bias literature as a known-noise-schedule
   diffusion process: option order (Shi et al., 2025), rubric paraphrase
   (SPUQ), criterion reorder + score-ID format (Chen et al., 2025), sampling
   temperature (Rating Roulette), exemplar resample, frame shuffle.
2. **Single-step Tweedie posterior-mean denoiser** (analytical via Gaussian
   KDE; ablated against a tiny learned MLP). Produces a calibrated point
   estimate **plus posterior variance** at no extra training cost.
3. **Ordinal-boundary-adjusted conformal intervals** (Sheng et al., EMNLP
   2025), studentized by the per-item Tweedie posterior σ̂. Adaptive widths
   without a separate quantile model.
4. **Eval-of-eval harness** that reports κ / α / Pearson / Spearman / ECE /
   Brier *and* per-bias-source robustness deltas + 5-seed stochastic
   stability — first-class outputs, not afterthoughts.

## Repository layout

```
diffujudge/
  config.py                  Pydantic config schema
  taxonomy/nhtsa.py          12-cat behavior + NHTSA pre-crash mapping
  data/                      LingoQA, CODA-LM, frame samplers, golden set, synthetic
  judges/
    base.py                  BaseJudge ABC
    mock_judge.py            deterministic bias-injecting mock (CI-safe)
    api_judge.py             LiteLLM (GPT-4o, Claude, Gemini) — closed-judge baseline
    vllm_judge.py            Qwen2.5-VL / InternVL2 / LLaVA-Critic via vLLM
    ensemble.py              multi-judge aggregator
  perturbations/             7-level forward cascade
  denoiser/
    tweedie.py               analytical KDE + Tweedie's formula
    learned.py               2-layer MLP (optional, requires torch)
  conformal/ordinal.py       boundary-adjusted split-conformal
  metrics/                   κ, α, ECE, Brier, bias deltas, stochastic stability
  eval/harness.py            eval-of-eval report builder
  pipeline.py                end-to-end orchestrator
  cli.py                     `diffujudge` Typer entry-point
nvidia/                      NVILA judge, CARLA scenario map, NIM deployment
dashboard/streamlit_app.py   reliability + interval + per-level dashboard
configs/                     Hydra yaml — default, smoke, lingoqa, coda_lm_stress
scripts/                     run_inference, eval, calibrate, build_golden_set, generate_paraphrases
tests/                       pytest suite (no GPU / no API)
```

## Quickstart — 30 seconds, CPU-only

```bash
git clone https://github.com/syedhumarahim/diffujudge-av
cd diffujudge-av
python -m venv .venv && source .venv/bin/activate
pip install -e .

# End-to-end on synthetic AV-flavored data with three mock judges:
python scripts/run_inference.py --dataset synthetic --n 50

# Run the eval-of-eval harness:
python scripts/eval.py --run-dir outputs/<fingerprint>

# Browse the calibration dashboard:
diffujudge dashboard --run-dir outputs/<fingerprint>
```

## Running with real VLMs

Single H100 (or 2× RTX 4090):

```bash
pip install -e ".[vllm,calibration,dashboard]"

python scripts/run_inference.py \
    --dataset lingoqa --data-dir data/lingoqa --n 200 \
    --backend vllm \
    --judges Qwen/Qwen2.5-VL-7B-Instruct,OpenGVLab/InternVL2-8B,lmms-lab/llava-critic-7b
```

API-fallback ensemble (no GPU, requires keys in `.env`):

```bash
pip install -e ".[api,calibration]"
python scripts/run_inference.py \
    --dataset synthetic --n 200 \
    --backend api \
    --judges gpt-4o-mini,claude-3-5-haiku-20241022,gemini-1.5-flash
```

## Evaluation methodology

### The 7-level forward perturbation cascade

| t | Operator | Bias source | Reference |
|---|---|---|---|
| 1 | option swap | position bias | Shi et al., IJCNLP-AACL 2025 |
| 2 | rubric paraphrase | prompt sensitivity | SPUQ — arXiv 2403.02509 |
| 3 | criterion reorder | rubric-order bias | Chen et al., arXiv 2506.22316 |
| 4 | score-ID format swap | scoring-ID bias | Chen et al., arXiv 2506.22316 |
| 5 | temperature noise | self-inconsistency | Thakur et al., arXiv 2510.27106 |
| 6 | exemplar resample | few-shot variance | classical |
| 7 | frame shuffle | temporal robustness (video) | this work |

Each level emits *k* samples (default 3), giving N×k score observations per
item. The pool is then denoised via Tweedie's formula and wrapped in a
conformal interval.

### Tweedie posterior-mean

Given y = x + ε with ε ~ N(0, σ²), Tweedie's identity yields

    E[x | y]   = y + σ² ∇ log p(y)
    Var[x | y] = σ² + σ⁴ ∇² log p(y)

p(y) is estimated with a Gaussian KDE over the N×k samples; per-level σ_t² is
the within-bucket variance; pooling is precision-weighted before the Tweedie
correction. Reference: Manor & Michaeli, ICLR 2024 (arXiv 2309.13598).

### Ordinal-boundary-adjusted conformal

Studentized split-conformal with the per-item Tweedie posterior std as the
nonconformity scaler. Lower / upper interval edges are snapped to the nearest
ordinal class boundary so the interval has a meaningful ordinal
interpretation. Reference: Sheng et al., EMNLP 2025 (arXiv 2509.18658).

### Three-tier golden set

  Tier 1 — Lingo-Judge anchors with ≥ 0.8 confidence (free).
  Tier 2 — Supermajority synthetic (GPT-4o + Claude + Gemini, κ > 0.8).
  Tier 3 — 30–50 manual hand-labels on hardest cases (corner-cut-ins at
  night, occluded VRUs, ambiguous near-misses). See `docs/annotation_guide.md`.

### Eval-of-eval metrics

- Cohen's κ (quadratic-weighted)
- Krippendorff's α (ordinal)
- Pearson / Spearman / Kendall τ
- ECE, MCE, Brier
- Position / verbosity / scoring-ID bias deltas
- 5-seed stochastic stability
- Conformal coverage (target ≥ 1 − α) + mean interval width

## Reproducibility

- Pydantic-typed configs, fingerprinted into the output path.
- Deterministic per-(item, level, k) seeds.
- JSONL streaming with per-record `fsync` so a crash leaves a valid prefix.
- Tests run on CPU without API keys.

## References

The methodology section above cites the primary works directly. A complete
bibliography of the 30+ papers that ground this design lives in
[`docs/methodology.md`](docs/methodology.md).

## License

MIT. See `LICENSE`.
