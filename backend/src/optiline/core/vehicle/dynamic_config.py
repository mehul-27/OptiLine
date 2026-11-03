"""Configuration for the M3.1 dynamic single-track vehicle model."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

DYNAMIC_VEHICLE_SCHEMA_VERSION = 1
IntegrationMethod = Literal["semi_implicit_euler", "rk4"]


class DynamicVehicleConfig(BaseModel):
    """Dynamic single-track parameters in SI units.

    The rear axle is the reference point. The CG is ``rear_axle_distance_m``
    ahead of it; the front axle is ``front_axle_distance_m`` ahead of the CG.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictInt
    name: str = Field(min_length=1)
    model_type: Literal["dynamic_single_track"]
    mass_kg: float = Field(gt=0.0, allow_inf_nan=False)
    yaw_inertia_kg_m2: float = Field(gt=0.0, allow_inf_nan=False)
    front_axle_distance_m: float = Field(gt=0.0, allow_inf_nan=False)
    rear_axle_distance_m: float = Field(gt=0.0, allow_inf_nan=False)
    front_cornering_stiffness_n_per_rad: float = Field(gt=0.0, allow_inf_nan=False)
    rear_cornering_stiffness_n_per_rad: float = Field(gt=0.0, allow_inf_nan=False)
    max_steering_angle_rad: float = Field(
        gt=0.0,
        lt=math.pi / 2.0,
        allow_inf_nan=False,
    )
    max_acceleration_mps2: float = Field(gt=0.0, allow_inf_nan=False)
    max_braking_mps2: float = Field(gt=0.0, allow_inf_nan=False)
    max_speed_mps: float = Field(gt=0.0, allow_inf_nan=False)
    low_speed_transition_start_mps: float = Field(gt=0.0, allow_inf_nan=False)
    low_speed_transition_end_mps: float = Field(gt=0.0, allow_inf_nan=False)

    @property
    def wheelbase_m(self) -> float:
        return self.front_axle_distance_m + self.rear_axle_distance_m

    @model_validator(mode="after")
    def validate_config(self) -> DynamicVehicleConfig:
        if self.schema_version != DYNAMIC_VEHICLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported dynamic vehicle schema_version: {self.schema_version}")
        if self.low_speed_transition_end_mps <= self.low_speed_transition_start_mps:
            raise ValueError(
                "low_speed_transition_end_mps must be greater than the start threshold",
            )
        if self.low_speed_transition_end_mps > self.max_speed_mps:
            raise ValueError("low_speed_transition_end_mps must not exceed max_speed_mps")
        return self


class DynamicSimulationConfig(BaseModel):
    """Fixed-step integration settings for the dynamic model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dt_s: float = Field(gt=0.0, allow_inf_nan=False)
    integration_method: IntegrationMethod
