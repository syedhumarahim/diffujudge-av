# NVIDIA AV Eval — vocabulary alignment

This document is a one-page crib sheet showing how each phrase in NVIDIA's
AV Eval job description maps onto a concrete artifact in this repository.

| Phrase from JD | Artifact in this repo |
|---|---|
| "Learned evaluation" | The `JudgeEnsemble` of three open VLMs (`diffujudge.judges.vllm_judge.VLLMJudge`) — Qwen2.5-VL-7B + InternVL2-8B + LLaVA-Critic-7B. |
| "Evaluation-of-evaluation" | `diffujudge.eval.EvalOfEvalHarness` — first-class κ / α / ECE / Brier / bias deltas / stochastic stability over the *judges*, not the model under test. |
| "Golden-set framework" | Three-tier golden set in `diffujudge.data.golden_set`: Lingo-Judge anchors + supermajority synthetic + manual hand-labels. |
| "Calibration loop" | `OrdinalBoundaryConformal.fit / .predict / .evaluate` — coverage at α=0.10, mean width, adaptive per-item scaling. |
| "Behavior taxonomy / cut-ins / hard braking / VRU / lane-keeping" | `diffujudge.taxonomy.nhtsa.BEHAVIOR_CATEGORIES` — 12-category schema aligned with NHTSA pre-crash typology (Najm et al., 2007/2019). |
| "Safety-critical event detection" | `BehaviorCategory.is_safety_critical`, used as the stratification axis in the eval harness. |
| "OpenSCENARIO / CARLA Leaderboard" | `nvidia/carla_scenarios.py` — many-to-many map between the 12-cat taxonomy, NHTSA IDs, and CARLA Leaderboard 2.0 scenarios. |
| "NVIDIA stack / NIM / NeMo" | `nvidia/nvila_integration.py` (NVILA-8B as a `BaseJudge`) and `nvidia/nim_deployment.md` (drop-in NIM endpoint via `VLLMJudge(server_url=…)`). |
| "Conformal prediction / uncertainty" | Tweedie posterior variance (`diffujudge.denoiser.tweedie`) studentizing the conformal nonconformity score. |
| "Inter-rater reliability" | `cohen_kappa`, `fleiss_kappa`, `krippendorff_alpha` in `diffujudge.metrics.agreement`. |

The intent is fluency, not gaming: every phrase corresponds to a
real-and-tested module.
