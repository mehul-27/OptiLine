"""Planning limits and provenance for the M4.1 baseline planner."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from optiline.core.vehicle.dynamic_config import DynamicVehicleConfig

PLANNING_SCHEMA_VERSION = 1
ILLUSTRATIVE_LATERAL_ACCELERATION_MPS2 = 4.0


class PlanningLimits(BaseModel):
    """Authoritative limits for baseline path-speed planning.

    The speed, acceleration and braking values are snapshotted from the
    authoritative dynamic vehicle configuration by
    :meth:`from_dynamic_vehicle_config`. The lateral limit is an independent
    planning constraint supplied explicitly; it is not derived from tyre
    stiffness or presented as a measured grip limit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictInt
    max_speed_mps: float = Field(gt=0.0, allow_inf_nan=False)
    max_longitudinal_acceleration_mps2: float = Field(gt=0.0, allow_inf_nan=False)
    max_braking_deceleration_mps2: float = Field(gt=0.0, allow_inf_nan=False)
    max_lateral_acceleration_mps2: float = Field(gt=0.0, allow_inf_nan=False)
    source_vehicle_name: str = Field(min_length=1)
    source_vehicle_schema_version: StrictInt
    lateral_limit_source: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_limits(self) -> PlanningLimits:
        if self.schema_version != PLANNING_SCHEMA_VERSION:
            raise ValueError(f"unsupported planning schema_version: {self.schema_version}")
        if self.source_vehicle_schema_version <= 0:
            raise ValueError("source_vehicle_schema_version must be positive")
        return self

    @classmethod
    def from_dynamic_vehicle_config(
        cls,
        config: DynamicVehicleConfig,
        *,
        max_lateral_acceleration_mps2: float = ILLUSTRATIVE_LATERAL_ACCELERATION_MPS2,
    ) -> PlanningLimits:
        """Snapshot vehicle speed/longitudinal limits and add a lateral limit.

        The vehicle configuration remains authoritative for maximum speed,
        maximum acceleration and maximum braking. The resulting
        ``PlanningLimits`` is the authoritative snapshot used by the planner
        and the independent validator.
        """

        return cls(
            schema_version=PLANNING_SCHEMA_VERSION,
            max_speed_mps=config.max_speed_mps,
            max_longitudinal_acceleration_mps2=config.max_acceleration_mps2,
            max_braking_deceleration_mps2=config.max_braking_mps2,
            max_lateral_acceleration_mps2=max_lateral_acceleration_mps2,
            source_vehicle_name=config.name,
            source_vehicle_schema_version=config.schema_version,
            lateral_limit_source=(
                "illustrative planning constraint; not a measured grip limit "
                "or universal vehicle property"
            ),
        )
