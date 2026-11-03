from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from optiline.core.vehicle.dynamic_config import (
    DYNAMIC_VEHICLE_SCHEMA_VERSION,
    DynamicSimulationConfig,
    DynamicVehicleConfig,
)


def make_dynamic_config(**overrides: object) -> DynamicVehicleConfig:
    data: dict[str, object] = {
        "schema_version": DYNAMIC_VEHICLE_SCHEMA_VERSION,
        "name": "dynamic test vehicle",
        "model_type": "dynamic_single_track",
        "mass_kg": 1500.0,
        "yaw_inertia_kg_m2": 2500.0,
        "front_axle_distance_m": 1.2,
        "rear_axle_distance_m": 1.6,
        "front_cornering_stiffness_n_per_rad": 80000.0,
        "rear_cornering_stiffness_n_per_rad": 80000.0,
        "max_steering_angle_rad": 0.5,
        "max_acceleration_mps2": 5.0,
        "max_braking_mps2": 10.0,
        "max_speed_mps": 50.0,
        "low_speed_transition_start_mps": 2.0,
        "low_speed_transition_end_mps": 4.0,
    }
    data.update(overrides)
    return DynamicVehicleConfig.model_validate(data)


def test_valid_dynamic_config_parses() -> None:
    config = make_dynamic_config()
    assert config.wheelbase_m == pytest.approx(2.8)
    assert config.low_speed_transition_start_mps == pytest.approx(2.0)


def test_dynamic_config_json_round_trip() -> None:
    config = make_dynamic_config()
    loaded = DynamicVehicleConfig.model_validate_json(config.model_dump_json())
    assert loaded == config


def test_missing_dynamic_field_fails() -> None:
    data = make_dynamic_config().model_dump()
    del data["mass_kg"]
    with pytest.raises(ValidationError):
        DynamicVehicleConfig.model_validate(data)


def test_unknown_dynamic_field_fails() -> None:
    data = make_dynamic_config().model_dump()
    data["unexpected"] = 1
    with pytest.raises(ValidationError):
        DynamicVehicleConfig.model_validate(data)


def test_invalid_dynamic_model_type_fails() -> None:
    with pytest.raises(ValidationError):
        make_dynamic_config(model_type="kinematic_bicycle")


@pytest.mark.parametrize("version", [0, 2, -1, True, 1.0])
def test_invalid_dynamic_schema_version_fails(version: object) -> None:
    with pytest.raises(ValidationError):
        make_dynamic_config(schema_version=version)


@pytest.mark.parametrize(
    "field_name",
    [
        "mass_kg",
        "yaw_inertia_kg_m2",
        "front_axle_distance_m",
        "rear_axle_distance_m",
        "front_cornering_stiffness_n_per_rad",
        "rear_cornering_stiffness_n_per_rad",
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_invalid_positive_dynamic_parameters_fail(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValidationError):
        make_dynamic_config(**{field_name: value})


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, math.pi / 2.0, math.pi / 2.0 + 0.001, math.nan, math.inf],
)
def test_invalid_dynamic_steering_limit_fails(value: float) -> None:
    with pytest.raises(ValidationError):
        make_dynamic_config(max_steering_angle_rad=value)


@pytest.mark.parametrize(
    "field_name",
    ["max_acceleration_mps2", "max_braking_mps2", "max_speed_mps"],
)
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_invalid_dynamic_limits_fail(field_name: str, value: float) -> None:
    with pytest.raises(ValidationError):
        make_dynamic_config(**{field_name: value})


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_invalid_low_speed_start_fails(value: float) -> None:
    with pytest.raises(ValidationError):
        make_dynamic_config(low_speed_transition_start_mps=value)


@pytest.mark.parametrize("value", [2.0, 1.0, 51.0, math.nan, math.inf])
def test_invalid_low_speed_end_fails(value: float) -> None:
    with pytest.raises(ValidationError):
        make_dynamic_config(low_speed_transition_end_mps=value)


@pytest.mark.parametrize("method", ["semi_implicit_euler", "rk4"])
def test_valid_simulation_config(method: str) -> None:
    config = DynamicSimulationConfig(dt_s=0.01, integration_method=method)  # type: ignore[arg-type]
    assert config.dt_s == pytest.approx(0.01)


@pytest.mark.parametrize("dt_s", [0.0, -0.1, math.nan, math.inf])
def test_invalid_simulation_timestep_fails(dt_s: float) -> None:
    with pytest.raises(ValidationError):
        DynamicSimulationConfig(dt_s=dt_s, integration_method="rk4")


def test_invalid_integration_method_fails() -> None:
    with pytest.raises(ValidationError):
        DynamicSimulationConfig(dt_s=0.01, integration_method="euler")  # type: ignore[arg-type]
