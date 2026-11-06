"""Versioned data contracts for the M4.1 baseline planner."""

from __future__ import annotations

import math
from dataclasses import dataclass

from optiline.core.planning.config import PLANNING_SCHEMA_VERSION, PlanningLimits

_CLOSURE_TOL = 1e-9
_UNIT_TOL = 1e-9


def _finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class PathSample:
    """One sample on a closed baseline path."""

    x_m: float
    y_m: float
    s_m: float
    tangent_x: float
    tangent_y: float
    curvature_1_per_m: float

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "s_m", "tangent_x", "tangent_y", "curvature_1_per_m"):
            _finite(getattr(self, name), name)
        if self.s_m < 0.0:
            raise ValueError("s_m must be non-negative")
        tangent_norm = math.hypot(self.tangent_x, self.tangent_y)
        if abs(tangent_norm - 1.0) > _UNIT_TOL:
            raise ValueError("path tangent must be a unit vector")


@dataclass(frozen=True)
class BaselinePath:
    """Closed, arc-length-parameterised baseline path.

    ``samples`` contains ``unique_sample_count + 1`` entries. The final entry
    repeats the first geometric point at arc length ``length_m``; it is not a
    zero-distance interval. The final interval from the penultimate sample to
    the repeated endpoint is part of the path.
    """

    schema_version: int
    source_track_id: str
    source_track_name: str
    length_m: float
    unique_sample_count: int
    closed: bool
    samples: tuple[PathSample, ...]

    def __post_init__(self) -> None:
        if self.schema_version != PLANNING_SCHEMA_VERSION:
            raise ValueError(f"unsupported path schema_version: {self.schema_version}")
        if not self.source_track_id or not self.source_track_name:
            raise ValueError("path source identity must not be empty")
        _finite(self.length_m, "length_m")
        if self.length_m <= 0.0:
            raise ValueError("length_m must be positive")
        if self.unique_sample_count < 3:
            raise ValueError("unique_sample_count must be at least three")
        if not self.closed:
            raise ValueError("M4.1 requires a closed path")
        if len(self.samples) != self.unique_sample_count + 1:
            raise ValueError("closed path must contain unique_sample_count + 1 samples")
        first = self.samples[0]
        last = self.samples[-1]
        if abs(first.x_m - last.x_m) > _CLOSURE_TOL or abs(first.y_m - last.y_m) > _CLOSURE_TOL:
            raise ValueError("path start and end positions must match")
        if abs(first.s_m) > _CLOSURE_TOL:
            raise ValueError("path must start at s=0")
        if abs(last.s_m - self.length_m) > _CLOSURE_TOL:
            raise ValueError("path final sample must be at length_m")
        previous = 0.0
        for sample in self.samples:
            if sample.s_m <= previous and sample is not first:
                raise ValueError("path arc length must increase strictly")
            previous = sample.s_m


@dataclass(frozen=True)
class SpeedProfileSample:
    """Speed and derived quantities at one path sample."""

    s_m: float
    speed_mps: float
    curvature_1_per_m: float
    lateral_acceleration_mps2: float

    def __post_init__(self) -> None:
        for name in (
            "s_m",
            "speed_mps",
            "curvature_1_per_m",
            "lateral_acceleration_mps2",
        ):
            _finite(getattr(self, name), name)
        if self.s_m < 0.0:
            raise ValueError("s_m must be non-negative")
        if self.speed_mps <= 0.0:
            raise ValueError("speed_mps must be positive for a flying lap")
        if self.lateral_acceleration_mps2 < 0.0:
            raise ValueError("lateral_acceleration_mps2 must be non-negative")


@dataclass(frozen=True)
class SpeedProfile:
    """Closed flying-lap speed profile.

    ``samples`` has one entry per path sample (including the repeated endpoint).
    ``interval_longitudinal_accelerations_mps2`` has one value per path
    interval, including the final closure interval.
    """

    schema_version: int
    source_track_id: str
    path_length_m: float
    unique_sample_count: int
    closed: bool
    samples: tuple[SpeedProfileSample, ...]
    interval_longitudinal_accelerations_mps2: tuple[float, ...]
    lap_time_s: float
    limits: PlanningLimits

    def __post_init__(self) -> None:
        if self.schema_version != PLANNING_SCHEMA_VERSION:
            raise ValueError(f"unsupported profile schema_version: {self.schema_version}")
        if not self.source_track_id:
            raise ValueError("profile source identity must not be empty")
        _finite(self.path_length_m, "path_length_m")
        if self.path_length_m <= 0.0:
            raise ValueError("path_length_m must be positive")
        if self.unique_sample_count < 3:
            raise ValueError("unique_sample_count must be at least three")
        if not self.closed:
            raise ValueError("M4.1 requires a closed profile")
        if len(self.samples) != self.unique_sample_count + 1:
            raise ValueError("profile must contain unique_sample_count + 1 samples")
        if len(self.interval_longitudinal_accelerations_mps2) != self.unique_sample_count:
            raise ValueError("profile must contain one interval acceleration per interval")
        first = self.samples[0]
        last = self.samples[-1]
        if abs(first.s_m) > _CLOSURE_TOL:
            raise ValueError("profile must start at s=0")
        if abs(last.s_m - self.path_length_m) > _CLOSURE_TOL:
            raise ValueError("profile final sample must be at path_length_m")
        if abs(first.speed_mps - last.speed_mps) > _CLOSURE_TOL:
            raise ValueError("profile start and end speeds must match")
        previous = 0.0
        for sample in self.samples:
            if sample.s_m <= previous and sample is not first:
                raise ValueError("profile arc length must increase strictly")
            previous = sample.s_m
        for acceleration in self.interval_longitudinal_accelerations_mps2:
            _finite(acceleration, "interval longitudinal acceleration")
        _finite(self.lap_time_s, "lap_time_s")
        if self.lap_time_s <= 0.0:
            raise ValueError("lap_time_s must be positive")
        if not isinstance(self.limits, PlanningLimits):
            raise ValueError("limits must be PlanningLimits")
