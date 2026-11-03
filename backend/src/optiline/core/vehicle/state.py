"""Vehicle state and control inputs for the kinematic bicycle model."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleState:
    """Vehicle state at the rear-axle reference point in SI units.

    ``heading_rad`` is measured counter-clockwise from the world +x axis.
    Speed is non-negative; reverse motion is not part of this model.
    """

    x_m: float
    y_m: float
    heading_rad: float
    speed_mps: float

    def __post_init__(self) -> None:
        for field_name in ("x_m", "y_m", "heading_rad", "speed_mps"):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.speed_mps < 0.0:
            raise ValueError("speed_mps must be non-negative")


@dataclass(frozen=True)
class VehicleControl:
    """Instantaneous steering angle and longitudinal acceleration."""

    steering_angle_rad: float
    longitudinal_acceleration_mps2: float

    def __post_init__(self) -> None:
        for field_name in ("steering_angle_rad", "longitudinal_acceleration_mps2"):
            if not math.isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite")
