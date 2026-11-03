"""Configuration for OptiLine's first vehicle model."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

VEHICLE_SCHEMA_VERSION = 1


class VehicleConfig(BaseModel):
    """Kinematic bicycle configuration in SI units.

    The reference point is the rear axle. Steering is an instantaneous control
    angle at the front wheel; reverse motion and steering actuator dynamics are
    not modelled in this version.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictInt
    name: str = Field(min_length=1)
    model_type: Literal["kinematic_bicycle"]
    wheelbase_m: float = Field(gt=0.0, allow_inf_nan=False)
    max_steering_angle_rad: float = Field(
        gt=0.0,
        lt=math.pi / 2.0,
        allow_inf_nan=False,
    )
    max_acceleration_mps2: float = Field(gt=0.0, allow_inf_nan=False)
    max_braking_mps2: float = Field(gt=0.0, allow_inf_nan=False)
    max_speed_mps: float = Field(gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_config(self) -> VehicleConfig:
        if self.schema_version != VEHICLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported vehicle schema_version: {self.schema_version}")
        return self
