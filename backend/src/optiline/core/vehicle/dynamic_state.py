"""Six-state vehicle state for the M3.1 dynamic single-track model."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicVehicleState:
    """Rear-axle dynamic state in SI units.

    ``v_x_mps`` and ``v_y_mps`` are body-frame velocities at the rear axle:
    +x forward and +y left. ``heading_rad`` is counter-clockwise from world +x.
    ``yaw_rate_radps`` is positive counter-clockwise. Reverse motion is not
    supported in M3.1.
    """

    x_m: float
    y_m: float
    heading_rad: float
    v_x_mps: float
    v_y_mps: float
    yaw_rate_radps: float

    def __post_init__(self) -> None:
        for field_name in (
            "x_m",
            "y_m",
            "heading_rad",
            "v_x_mps",
            "v_y_mps",
            "yaw_rate_radps",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.v_x_mps < 0.0:
            raise ValueError("v_x_mps must be non-negative")

    @property
    def speed_mps(self) -> float:
        return math.hypot(self.v_x_mps, self.v_y_mps)
