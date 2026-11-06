from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from optiline.core.planning import (
    ILLUSTRATIVE_LATERAL_ACCELERATION_MPS2,
    PLANNING_SCHEMA_VERSION,
    PlanningLimits,
    RacingLineOptimisationConfig,
)
from optiline.core.vehicle.dynamic_config import DynamicVehicleConfig


def test_planning_limits_snapshot_vehicle_values(
    vehicle_config: DynamicVehicleConfig,
) -> None:
    limits = PlanningLimits.from_dynamic_vehicle_config(vehicle_config)
    assert limits.schema_version == PLANNING_SCHEMA_VERSION
    assert limits.max_speed_mps == pytest.approx(vehicle_config.max_speed_mps)
    assert limits.max_longitudinal_acceleration_mps2 == pytest.approx(
        vehicle_config.max_acceleration_mps2,
    )
    assert limits.max_braking_deceleration_mps2 == pytest.approx(
        vehicle_config.max_braking_mps2,
    )
    assert limits.max_lateral_acceleration_mps2 == pytest.approx(
        ILLUSTRATIVE_LATERAL_ACCELERATION_MPS2,
    )
    assert limits.source_vehicle_name == vehicle_config.name
    assert limits.source_vehicle_schema_version == vehicle_config.schema_version
    assert "not a measured grip limit" in limits.lateral_limit_source


def test_planning_lateral_limit_is_configurable(
    vehicle_config: DynamicVehicleConfig,
) -> None:
    limits = PlanningLimits.from_dynamic_vehicle_config(
        vehicle_config,
        max_lateral_acceleration_mps2=3.5,
    )
    assert limits.max_lateral_acceleration_mps2 == pytest.approx(3.5)


def test_planning_limits_are_an_immutable_snapshot(
    vehicle_config: DynamicVehicleConfig,
) -> None:
    limits = PlanningLimits.from_dynamic_vehicle_config(vehicle_config)
    changed_vehicle = vehicle_config.model_copy(update={"max_speed_mps": 10.0})
    assert changed_vehicle.max_speed_mps == pytest.approx(10.0)
    assert limits.max_speed_mps == pytest.approx(vehicle_config.max_speed_mps)
    with pytest.raises(ValidationError):
        limits.max_speed_mps = 5.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_name",
    [
        "max_speed_mps",
        "max_longitudinal_acceleration_mps2",
        "max_braking_deceleration_mps2",
        "max_lateral_acceleration_mps2",
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_invalid_planning_limits_fail(
    planning_limits: PlanningLimits,
    field_name: str,
    value: float,
) -> None:
    data = planning_limits.model_dump()
    data[field_name] = value
    with pytest.raises(ValidationError):
        PlanningLimits.model_validate(data)


def test_invalid_planning_schema_and_source_fail(
    planning_limits: PlanningLimits,
) -> None:
    data = planning_limits.model_dump()
    data["schema_version"] = 2
    with pytest.raises(ValidationError):
        PlanningLimits.model_validate(data)

    data = planning_limits.model_dump()
    data["source_vehicle_schema_version"] = 0
    with pytest.raises(ValidationError):
        PlanningLimits.model_validate(data)

    data = planning_limits.model_dump()
    data["lateral_limit_source"] = ""
    with pytest.raises(ValidationError):
        PlanningLimits.model_validate(data)

    data = planning_limits.model_dump()
    data["unexpected"] = 1
    with pytest.raises(ValidationError):
        PlanningLimits.model_validate(data)


def test_racing_line_optimisation_config_defaults_and_validation() -> None:
    config = RacingLineOptimisationConfig()
    assert config.control_point_count == 24
    assert config.boundary_margin_m == pytest.approx(0.5)
    assert config.max_iterations == 200
    assert config.ftol == pytest.approx(1e-9)
    assert config.lap_time_improvement_threshold_s == pytest.approx(0.01)

    with pytest.raises(ValidationError):
        RacingLineOptimisationConfig(control_point_count=3)
    with pytest.raises(ValidationError):
        RacingLineOptimisationConfig(control_point_count=129)
    with pytest.raises(ValidationError):
        RacingLineOptimisationConfig(boundary_margin_m=0.0)
    with pytest.raises(ValidationError):
        RacingLineOptimisationConfig(boundary_margin_m=math.nan)
    with pytest.raises(ValidationError):
        RacingLineOptimisationConfig(max_iterations=0)
    with pytest.raises(ValidationError):
        RacingLineOptimisationConfig(ftol=0.0)
    with pytest.raises(ValidationError):
        RacingLineOptimisationConfig(lap_time_improvement_threshold_s=-0.01)
    with pytest.raises(ValidationError):
        RacingLineOptimisationConfig(lap_time_improvement_threshold_s=math.nan)
    with pytest.raises(ValidationError):
        RacingLineOptimisationConfig(lap_time_improvement_threshold_s=math.inf)
    with pytest.raises(ValidationError):
        RacingLineOptimisationConfig(schema_version=2)
    with pytest.raises(ValidationError):
        RacingLineOptimisationConfig(unexpected=1)  # type: ignore[call-arg]
