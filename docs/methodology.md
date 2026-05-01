# Methodology

This document expands the README's brief methodology section into a tactical,
citation-anchored deep dive. It is structured to lift directly into a Towards
Data Science post or a 4-page technical report.

---

## 1. The framing — Score Diffusion Judging (SDJ)

We treat the LLM/VLM judge's raw score `s̃_t` as a noisy observation of a
latent ideal rubric score `s_0` at noise level `t`, where `t` is controlled
by a structured perturbation schedule over:

  (i)   prompt paraphrases
  (ii)  rubric-criterion ordering
  (iii) score-ID format
  (iv)  sampling temperature
  (v)   reference-answer presentation
  (vi)  few-shot exemplar resample
  (vii) frame-order shuffle (video-only)

Running the judge `N` times across this schedule gives a *score distribution*
`p(s̃ | x, t)`. A single-step Tweedie denoiser then maps that distribution
back to a calibrated point estimate plus posterior variance.

The framing is novel: each piece (perturbation as noise, Tweedie as denoiser,
conformal as interval) has a 2024–25 citation, but the unified composition
applied to a safety-critical AV video benchmark is — to our knowledge —
unpublished.

## 2. Forward process — Calibrated Perturbation Cascade

Each of the 7 levels corresponds to one canonical bias source; the operator
is a deterministic function of `(view, seed, level)` so cascades are exactly
reproducible:

| t | Operator | Bias source | Citation |
|---|---|---|---|
| 1 | option swap | position bias | Shi et al., IJCNLP-AACL 2025 |
| 2 | rubric paraphrase | prompt sensitivity | Gao et al., SPUQ, arXiv 2403.02509 |
| 3 | criterion reorder | rubric-order bias | Chen et al., arXiv 2506.22316 |
| 4 | score-ID format swap | scoring-ID bias | Chen et al., arXiv 2506.22316 |
| 5 | temperature noise | self-inconsistency | Thakur et al., arXiv 2510.27106 |
| 6 | exemplar resample | few-shot variance | classical |
| 7 | frame shuffle | temporal robustness | this work (video-only) |

We additionally consider, but do not include in the headline run:

- **provenance / recency cues** (arXiv 2509.26072) — adversarial labels
  ("EXPERT", "old", etc.)
- **factor collapse** (arXiv 2509.20293) — semantically distinct rubric
  factors that judges fail to separate

These are stress-test extensions; see `tests/` for unit tests against the
operator interface.

## 3. Reverse process — Tweedie posterior-mean denoising

### 3.1 Closed form

For Gaussian noise y = x + ε, ε ~ N(0, σ²), Tweedie's identity (Robbins 1956;
Efron 2011) gives the posterior mean:

    E[x | y] = y + σ² · ∇_y log p(y)

We estimate p(y) with a 1-D Gaussian KDE over the N×k perturbed samples per
item with bandwidth h. The KDE score function admits a clean
softmax-weighted-residual closed form:

    let w_i(y) ∝ exp(-(y - y_i)² / (2h²))
    ∇ log p̂(y) = (1 / h²) (μ_w(y) - y)

where μ_w is the softmax-weighted mean of {y_i}. Substituting:

    ŝ_0 = ȳ + (σ̂_ε² / h²) (μ_w(ȳ) - ȳ)

where ȳ is the precision-weighted mean of the samples (per-level σ_t² as
inverse-variance weights) and σ̂_ε² is the corresponding pooled noise
variance.

### 3.2 Posterior variance

The second-order Tweedie identity (Manor & Michaeli, ICLR 2024) yields:

    Var[x | y] = σ² + σ⁴ · ∇² log p(y)

with KDE Hessian:

    ∇² log p̂(y) = Var_w(y_i) / h⁴ - 1 / h²

This gives us posterior σ̂² **with no additional training**, which becomes the
nonconformity scaler in the conformal layer.

### 3.3 Ablation: learned MLP

A 2-layer 64-unit MLP with heteroscedastic Gaussian NLL on the calibration
slice (~120 items) provides the comparison baseline. Architecturally borrowed
from SiDyP (Cao et al., KDD 2025, arXiv 2505.19675), specialized to ordinal
scalar scores.

## 4. Calibration — ordinal-boundary-adjusted conformal

Following Sheng et al. (EMNLP 2025, arXiv 2509.18658) and the 2026 follow-up
on VLM judges:

1. Calibration set with golden labels.
2. Studentized nonconformity α_i = |y_i − ŝ_i| / max(σ̂_i, ε).
3. Finite-sample-corrected quantile q̂ = Quantile_⌈(n+1)(1−α)/n⌉(α).
4. Test interval [ŝ − q̂σ̂, ŝ + q̂σ̂], snapped to nearest ordinal boundary.

The studentization couples interval width to the Tweedie posterior variance,
so confident items get tight intervals "for free."

We use MAPIE 0.8.6 as a sanity-check; the implementation in
`diffujudge/conformal/ordinal.py` is dependency-free and produces identical
intervals up to floating-point rounding.

## 5. Eval-of-eval

The marketable contribution per the design's §2.6 (the "evaluation-of-
evaluation" subfield):

- κ / α: inter-rater reliability against the three-tier golden set.
- ECE / MCE / Brier: calibration on the score distribution before and after
  Tweedie.
- Bias deltas: |score(A,B) − score(B,A)| (position), spearman(score, length)
  partialed on gold (verbosity), |score_arabic − score_roman| (scoring-ID).
- Stochastic stability: per-item std-dev across 5 seeds — the canonical
  "rating roulette" diagnostic.
- Conformal coverage at α = 0.10 + mean interval width.

All metrics dump to `outputs/<fingerprint>/eval_report.json` for the
dashboard.

## 6. Targets

| Metric | Baseline (single judge) | Target (DiffuJudge-AV) |
|---|---|---|
| Cohen's κ | ~0.45–0.55 | ≥ 0.65 |
| ECE | ~0.12 | ≤ 0.05 |
| Conformal coverage @ α=0.10 | n/a | ≥ 0.88 |
| Position-swap delta | ~0.6 | ≤ 0.3 |
| 5-seed std-dev | ~0.4 | ≤ 0.15 |
| Cost / item | ~$0.02 (GPT-4o) | ≤ $0.001 (open VLMs on H100) |

These match the design's §8 headline numbers; achievement is a function of
data quality and judge ensemble — the framework provides the path.

## 7. References

- Shi et al., *Judging the Judges: A Systematic Investigation of Position
  Bias in Pairwise Comparative Assessments*, IJCNLP-AACL 2025.
- Gao et al., *SPUQ — Perturbation-Based Uncertainty Quantification for
  Large Language Models*, arXiv 2403.02509.
- Chen et al., *Evaluating Scoring Bias in LLM-as-a-Judge*, arXiv 2506.22316.
- Thakur et al., *Rating Roulette: Self-Inconsistency in LLM-as-a-Judge*,
  arXiv 2510.27106.
- Manor & Michaeli, *Posterior-Mean Denoising via Tweedie's Formula*, ICLR
  2024 (arXiv 2309.13598).
- Sheng et al., *Analyzing Uncertainty of LLM-as-a-Judge with Conformal
  Prediction*, EMNLP 2025 (arXiv 2509.18658).
- Cao et al., *SiDyP — Simplex Label Diffusion with Dynamic Prior*, KDD 2025
  (arXiv 2505.19675).
- Marcu et al., *LingoQA — Visual Question Answering for Autonomous
  Driving*, ECCV 2024.
- Najm et al., *Pre-Crash Scenario Typology for Crash Avoidance Research*,
  NHTSA DOT HS 810 767 (2007, revised 2019).
- Gu et al., *A Survey on LLM-as-a-Judge*, arXiv 2411.15594 (v6 Oct 2025).
