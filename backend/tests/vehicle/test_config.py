from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from optiline.core.vehicle.config import VEHICLE_SCHEMA_VERSION, VehicleConfig


def make_config(**overrides: object) -> VehicleConfig:
    data: dict[str, object] = {
        "schema_version": VEHICLE_SCHEMA_VERSION,
        "name": "test vehicle",
        "model_type": "kinematic_bicycle",
        "wheelbase_m": 2.5,
        "max_steering_angle_rad": math.radians(30.0),
        "max_acceleration_mps2": 5.0,
        "max_braking_mps2": 10.0,
        "max_speed_mps": 50.0,
    }
    data.update(overrides)
    return VehicleConfig.model_validate(data)


def test_valid_config_parses() -> None:
    config = make_config()
    assert config.schema_version == VEHICLE_SCHEMA_VERSION
    assert config.wheelbase_m == pytest.approx(2.5)


def test_config_json_round_trip() -> None:
    config = make_config()
    loaded = VehicleConfig.model_validate_json(config.model_dump_json())
    assert loaded == config


def test_missing_required_field_fails() -> None:
    data = make_config().model_dump()
    del data["wheelbase_m"]
    with pytest.raises(ValidationError):
        VehicleConfig.model_validate(data)


def test_unknown_field_fails() -> None:
    data = make_config().model_dump()
    data["unexpected"] = 1
    with pytest.raises(ValidationError):
        VehicleConfig.model_validate(data)


def test_invalid_model_type_fails() -> None:
    with pytest.raises(ValidationError):
        make_config(model_type="dynamic_bicycle")


@pytest.mark.parametrize("version", [0, 2, -1, True, 1.0])
def test_invalid_schema_version_fails(version: object) -> None:
    with pytest.raises(ValidationError):
        make_config(schema_version=version)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_invalid_wheelbase_fails(value: float) -> None:
    with pytest.raises(ValidationError):
        make_config(wheelbase_m=value)


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, math.pi / 2.0, math.pi / 2.0 + 0.001, math.nan, math.inf],
)
def test_invalid_steering_limit_fails(value: float) -> None:
    with pytest.raises(ValidationError):
        make_config(max_steering_angle_rad=value)


@pytest.mark.parametrize(
    "field_name",
    ["max_acceleration_mps2", "max_braking_mps2", "max_speed_mps"],
)
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_invalid_non_negative_limits_fail(field_name: str, value: float) -> None:
    with pytest.raises(ValidationError):
        make_config(**{field_name: value})
