"""Versioned track schema for OptiLine."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Point2D = tuple[FiniteFloat, FiniteFloat]
SCHEMA_VERSION = 1

CoordinateSystem = Literal["x_right_y_up_ccw_rad"]
Units = Literal["m", "unspecified"]
CalibrationStatus = Literal["calibrated", "uncalibrated"]
CalibrationMethod = Literal["known_distance"]


class TrackModel(BaseModel):
    """Base model with the schema's shared strictness settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Boundary(TrackModel):
    """A closed polyline stored without repeating its first point."""

    points: list[Point2D] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_boundary(self) -> Boundary:
        points = self.points
        if points[0] == points[-1]:
            raise ValueError("boundary must not repeat the first point at the end")
        if len(set(points)) < 3:
            raise ValueError("boundary must contain at least three distinct points")
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            if point == next_point:
                raise ValueError("boundary must not contain duplicate consecutive points")
            if not (math.isfinite(point[0]) and math.isfinite(point[1])):
                raise ValueError("boundary points must be finite")
        return self


class TrackBoundaries(TrackModel):
    """Outer and inner track boundaries."""

    outer: Boundary
    inner: Boundary


class StartFinish(TrackModel):
    """Directed start/finish segment and the direction of travel."""

    line: tuple[Point2D, Point2D]
    forward: Point2D

    @model_validator(mode="after")
    def validate_start_finish(self) -> StartFinish:
        start, end = self.line
        if start == end:
            raise ValueError("start/finish line endpoints must be distinct")
        forward_norm = math.hypot(self.forward[0], self.forward[1])
        if forward_norm == 0.0:
            raise ValueError("start/finish forward vector must be non-zero")
        if not math.isclose(forward_norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("start/finish forward vector must be a unit vector")
        return self


class Calibration(TrackModel):
    """Calibration provenance for a track's world geometry."""

    status: CalibrationStatus
    method: CalibrationMethod | None = None
    scale_m_per_unit: float | None = Field(default=None, gt=0.0)
    rotation_rad: float | None = None
    translation: Point2D | None = None
    reference_points: tuple[Point2D, Point2D] | None = None
    reference_distance_m: float | None = Field(default=None, gt=0.0)
    source: str | None = None

    @model_validator(mode="after")
    def validate_calibration(self) -> Calibration:
        if self.status == "calibrated":
            required = (
                self.method,
                self.scale_m_per_unit,
                self.rotation_rad,
                self.translation,
                self.reference_points,
                self.reference_distance_m,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "calibrated track requires method, scale, rotation, translation, "
                    "reference points and reference distance"
                )
            if self.reference_points is not None and (
                self.reference_points[0] == self.reference_points[1]
            ):
                raise ValueError("calibration reference points must be distinct")
            if self.rotation_rad is not None and not math.isfinite(self.rotation_rad):
                raise ValueError("calibration rotation must be finite")
            if self.scale_m_per_unit is not None and not math.isfinite(self.scale_m_per_unit):
                raise ValueError("calibration scale must be finite")
            if self.reference_distance_m is not None and not math.isfinite(
                self.reference_distance_m,
            ):
                raise ValueError("calibration reference distance must be finite")
            return self

        if any(
            value is not None
            for value in (
                self.method,
                self.scale_m_per_unit,
                self.rotation_rad,
                self.translation,
                self.reference_points,
                self.reference_distance_m,
            )
        ):
            raise ValueError("uncalibrated track must not include transform values")
        return self


class Track(TrackModel):
    """Canonical persisted track inputs.

    Derived values such as length, width, curvature and validation results are
    computed on load and are deliberately not stored in this model.
    """

    schema_version: int
    track_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    coordinate_system: CoordinateSystem = "x_right_y_up_ccw_rad"
    units: Units
    boundaries: TrackBoundaries
    start_finish: StartFinish
    calibration: Calibration

    @model_validator(mode="after")
    def validate_track(self) -> Track:
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.calibration.status == "calibrated" and self.units != "m":
            raise ValueError("calibrated track must use units='m'")
        if self.calibration.status == "uncalibrated" and self.units != "unspecified":
            raise ValueError("uncalibrated track must use units='unspecified'")
        return self
