"""Cross-walk between our 12-category behavior taxonomy, NHTSA pre-crash IDs,
and CARLA Leaderboard 2.0 scenario routes.

Reference:
- CARLA Leaderboard 2.0 scenarios — https://leaderboard.carla.org/scenarios/
- ASAM OpenSCENARIO 1.x — behavior catalog
- NHTSA Najm et al. 2007/2019 pre-crash typology

The mapping is intentionally many-to-many; both sides are coarser than full
behavioral specifications.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CarlaScenarioMapping:
    behavior_code: str
    nhtsa_scenario: int | None
    carla_scenarios: tuple[str, ...]
    openscenario_phenomenon: str


MAPPINGS: tuple[CarlaScenarioMapping, ...] = (
    CarlaScenarioMapping("cut_in", 22, ("LeadCutIn", "VehicleCutIn"), "LaneChangeCondition"),
    CarlaScenarioMapping("hard_brake", 21, ("HardBrakeRoute", "FollowLeadingVehicleHardBrake"), "RelativeSpeedCondition"),
    CarlaScenarioMapping("lane_keep_dev", 14, ("StationaryObjectCrossing",), "LaneOffsetCondition"),
    CarlaScenarioMapping("left_turn_oncoming", 2, ("LeftTurn", "OppositeVehicleRunningRedLight"), "LaneChangeCondition"),
    CarlaScenarioMapping("right_turn_crossing", 3, ("RightTurn",), "LaneChangeCondition"),
    CarlaScenarioMapping("vru_conflict", 11, ("DynamicObjectCrossing", "PedestrianCrossing"), "DistanceCondition"),
    CarlaScenarioMapping("tailgating", 20, ("FollowLeadingVehicle",), "TimeHeadwayCondition"),
    CarlaScenarioMapping("yield_failure", 4, ("RunningRedLight", "RunningStopSign"), "ReachPositionCondition"),
    CarlaScenarioMapping("merge", 24, ("HighwayCutIn", "MergerIntoSlowTraffic"), "LaneChangeCondition"),
    CarlaScenarioMapping("lane_change_unsafe", 23, ("LaneChange",), "LaneChangeCondition"),
    CarlaScenarioMapping("control_loss", 1, ("ControlLoss",), "RoadCondition"),
)


_BY_CODE = {m.behavior_code: m for m in MAPPINGS}


def carla_for(behavior_code: str) -> tuple[str, ...]:
    return _BY_CODE[behavior_code].carla_scenarios if behavior_code in _BY_CODE else ()


def nhtsa_for(behavior_code: str) -> int | None:
    return _BY_CODE[behavior_code].nhtsa_scenario if behavior_code in _BY_CODE else None
