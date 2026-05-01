# Tier 3 manual annotation guide

This guide governs the 30–50-item manual gold set used as Tier 3 of the
three-tier golden set in `diffujudge.data.golden_set`.

The purpose of Tier 3 is **expert-anchored ground truth on the hardest cases**
— the items where automated Tier 1 / Tier 2 labels disagree or where the
event taxonomy is genuinely ambiguous.

## Selection criteria

Pick items that satisfy **at least one** of the following:

1. **Tier 1 ≠ Tier 2.** Lingo-Judge confidence ≥ 0.8 but supermajority
   synthetic disagrees by ≥ 1 score-class.
2. **Corner-case taxonomy hit.** Cut-ins at night, occluded VRUs, ambiguous
   near-misses, control-loss recoveries, unprotected lefts with multi-vehicle
   oncoming.
3. **High judge variance** in the perturbation cascade (top quartile of
   per-item std-dev across the seven perturbation levels).

## Annotation procedure

For each item:

1. Watch the full clip — don't rely on extracted frames alone.
2. Identify the dominant **safety-critical event** (or `no_conflict`) using
   the 12-category behavior taxonomy in `diffujudge/taxonomy/nhtsa.py`.
3. Score the candidate answer on a 1–5 ordinal scale **against the rubric
   below**, not against your gut.
4. Record the score, the chosen behavior code, and a one-sentence rationale.

## Rubric

| Score | Meaning |
|---|---|
| 5 | Identifies the correct event, names the safety-critical attribute (TTC, headway, lateral position), and is consistent with the reference. |
| 4 | Correct event identification + at least one safety-critical attribute named, but minor missing detail. |
| 3 | Correct event identification but generic / underspecified about safety-criticality. |
| 2 | Partial event identification — wrong agent, wrong direction, or wrong severity. |
| 1 | Hallucinated event or completely incorrect; or refuses to answer when an answer is determinable. |

## Inter-annotator reliability

If a second annotator can be recruited for ≥20 items, target Cohen's
quadratic-weighted κ ≥ 0.7 between annotators. Disagreements ≥ 2 score-classes
should trigger a re-watch and a discussion-to-consensus pass — record the
final consensus score and tag the item with `reconciled=true` in the meta
field.

## File format

Tier 3 lives in `data/golden/tier3_manual.jsonl`, one JSON object per line:

```json
{"item_id": "lingoqa_0042", "score": 4.0, "behavior_code": "cut_in", "rationale": "Silver sedan crosses lane line at ~1.2s TTC; reference matches; minor: headway not named."}
```

The `build_golden_set.py` script consumes this file directly.

## Mapping to NHTSA pre-crash scenarios

When labeling the behavior code, refer to
`diffujudge.taxonomy.NHTSA_PRECRASH_SCENARIOS`. Each behavior category in the
12-category schema is back-mapped to the canonical NHTSA scenario ID, and
`nvidia/carla_scenarios.py` extends the mapping to CARLA Leaderboard 2.0
scenarios so the manual labels are interoperable with simulator-generated
data.
