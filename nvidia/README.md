# `nvidia/` — production-stack integration recipes

This subfolder exists so reviewers from NVIDIA's AV Eval team can verify, in
~30 seconds, that the framework's primitives map cleanly onto the stack they
work in every day.

| File | What it shows |
|---|---|
| `nvila_integration.py` | Drop-in `BaseJudge` for **NVILA-8B**, NVIDIA's open VLM. Same interface as Qwen/InternVL — picks up the perturbation cascade and Tweedie denoiser unchanged. |
| `carla_scenarios.py` | Map between our 12-category behavior taxonomy and CARLA Leaderboard scenarios + NHTSA pre-crash IDs. Ready to seed `LingoQAItem.behavior_label` from CARLA scenario manifests. |
| `nim_deployment.md` | A short recipe for serving the judge ensemble as an OpenAI-compatible **NVIDIA NIM** endpoint, then pointing `VLLMJudge(server_url=...)` at it. |

The vocabulary alignment is intentional: this repo speaks "learned
evaluation," "evaluation-of-evaluation," "calibration loop," and "golden-set
framework" — the same phrases as the AV Eval job description.
