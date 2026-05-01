"""NHTSA-aligned behavior taxonomy for AV safety-critical video evaluation.

Reference: Najm et al., 2007, *Pre-Crash Scenario Typology for Crash Avoidance Research*
(NHTSA DOT HS 810 767), revised 2019. Aligned with ASAM OpenSCENARIO behavior labels
and CARLA Leaderboard scenarios.

The 12-category schema is the design's recommended starting taxonomy. Subset to
4–6 categories for a 24-hour build by setting `BEHAVIOR_CATEGORIES_SUBSET`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorCategory:
    code: str
    name: str
    nhtsa_scenario: int | None
    description: str
    is_safety_critical: bool


BEHAVIOR_CATEGORIES: tuple[BehaviorCategory, ...] = (
    BehaviorCategory(
        code="cut_in",
        name="Cut-in",
        nhtsa_scenario=22,
        description="Lateral incursion by another vehicle within ego's TTC envelope.",
        is_safety_critical=True,
    ),
    BehaviorCategory(
        code="hard_brake",
        name="Hard braking",
        nhtsa_scenario=21,
        description="Longitudinal deceleration > 0.4g; maps to lead-vehicle-decelerating.",
        is_safety_critical=True,
    ),
    BehaviorCategory(
        code="lane_keep_dev",
        name="Lane-keeping deviation",
        nhtsa_scenario=14,
        description="Lane-line crossing without signal; drift or unintentional departure.",
        is_safety_critical=False,
    ),
    BehaviorCategory(
        code="left_turn_oncoming",
        name="Unprotected left turn with oncoming traffic",
        nhtsa_scenario=2,
        description="Left-turn-across-path / opposite-direction conflict.",
        is_safety_critical=True,
    ),
    BehaviorCategory(
        code="right_turn_crossing",
        name="Right-turn with crossing traffic",
        nhtsa_scenario=3,
        description="Intersection right-turn into laterally crossing traffic.",
        is_safety_critical=True,
    ),
    BehaviorCategory(
        code="vru_conflict",
        name="VRU conflict",
        nhtsa_scenario=11,
        description="Vulnerable road user (pedestrian/cyclist) conflict, incl. jaywalking.",
        is_safety_critical=True,
    ),
    BehaviorCategory(
        code="tailgating",
        name="Following too close / tailgating",
        nhtsa_scenario=20,
        description="Time headway THW < 1.0s.",
        is_safety_critical=True,
    ),
    BehaviorCategory(
        code="yield_failure",
        name="Yield-failure / stop-sign violation",
        nhtsa_scenario=4,
        description="Failure to stop or yield right-of-way.",
        is_safety_critical=True,
    ),
    BehaviorCategory(
        code="merge",
        name="Merge / on-ramp interaction",
        nhtsa_scenario=24,
        description="Merging onto highway or interaction at on-ramp.",
        is_safety_critical=False,
    ),
    BehaviorCategory(
        code="lane_change_unsafe",
        name="Lane change without sufficient gap",
        nhtsa_scenario=23,
        description="Lateral move into adjacent lane without adequate gap.",
        is_safety_critical=True,
    ),
    BehaviorCategory(
        code="control_loss",
        name="Control-loss recovery",
        nhtsa_scenario=1,
        description="Loss of vehicle control with subsequent recovery.",
        is_safety_critical=True,
    ),
    BehaviorCategory(
        code="no_conflict",
        name="Near-miss / no-conflict",
        nhtsa_scenario=None,
        description="Negative class — nominal driving with no safety-critical event.",
        is_safety_critical=False,
    ),
)


BEHAVIOR_CATEGORIES_SUBSET: tuple[str, ...] = (
    "cut_in",
    "hard_brake",
    "vru_conflict",
    "left_turn_oncoming",
    "tailgating",
    "no_conflict",
)


NHTSA_PRECRASH_SCENARIOS: dict[int, str] = {
    1: "Control loss without prior vehicle action",
    2: "Vehicle(s) turning at non-signalized junctions",
    3: "Straight crossing paths at non-signalized junctions",
    4: "Vehicle(s) failure-to-stop at non-signalized junctions",
    11: "Pedestrian / cyclist crossing roadway",
    14: "Single-vehicle road departure",
    20: "Lead vehicle stopped",
    21: "Lead vehicle decelerating",
    22: "Vehicle changing lanes — same direction",
    23: "Vehicle turning — same direction",
    24: "Vehicle entering roadway from driveway",
}


_CATEGORY_BY_CODE = {c.code: c for c in BEHAVIOR_CATEGORIES}


def classify_label(code: str) -> BehaviorCategory:
    if code not in _CATEGORY_BY_CODE:
        raise KeyError(f"Unknown behavior code '{code}'. Known: {list(_CATEGORY_BY_CODE)}")
    return _CATEGORY_BY_CODE[code]
