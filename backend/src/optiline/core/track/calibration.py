"""Similarity transforms and track calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from optiline.core.track.schema import (
    Boundary,
    Calibration,
    Point2D,
    StartFinish,
    Track,
    TrackBoundaries,
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SimilarityTransform:
    """A uniform scale, rotation and translation in the plane."""

    scale: float
    rotation_rad: float
    translation: tuple[float, float]

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("calibration scale must be finite and greater than zero")
        if not math.isfinite(self.rotation_rad):
            raise ValueError("calibration rotation must be finite")
        if len(self.translation) != 2:
            raise ValueError("calibration translation must contain two values")
        if not (math.isfinite(self.translation[0]) and math.isfinite(self.translation[1])):
            raise ValueError("calibration translation must be finite")

    def apply(self, points: FloatArray) -> FloatArray:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (n, 2)")
        cos_theta = math.cos(self.rotation_rad)
        sin_theta = math.sin(self.rotation_rad)
        x = points[:, 0]
        y = points[:, 1]
        transformed_x = self.scale * (cos_theta * x - sin_theta * y) + self.translation[0]
        transformed_y = self.scale * (sin_theta * x + cos_theta * y) + self.translation[1]
        return np.column_stack((transformed_x, transformed_y))

    def apply_direction(self, direction: Point2D) -> tuple[float, float]:
        cos_theta = math.cos(self.rotation_rad)
        sin_theta = math.sin(self.rotation_rad)
        x, y = direction
        return (
            cos_theta * x - sin_theta * y,
            sin_theta * x + cos_theta * y,
        )

    def inverse(self, points: FloatArray) -> FloatArray:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (n, 2)")
        translated_x = (points[:, 0] - self.translation[0]) / self.scale
        translated_y = (points[:, 1] - self.translation[1]) / self.scale
        cos_theta = math.cos(-self.rotation_rad)
        sin_theta = math.sin(-self.rotation_rad)
        return np.column_stack(
            (
                cos_theta * translated_x - sin_theta * translated_y,
                sin_theta * translated_x + cos_theta * translated_y,
            )
        )


def calibrate_from_known_distance(
    source_p1: Point2D,
    source_p2: Point2D,
    known_distance_m: float,
) -> SimilarityTransform:
    """Build a transform that maps ``source_p1`` to the origin and the segment to +x."""

    if not math.isfinite(known_distance_m) or known_distance_m <= 0.0:
        raise ValueError("known distance must be finite and positive")
    for point in (source_p1, source_p2):
        if not (math.isfinite(point[0]) and math.isfinite(point[1])):
            raise ValueError("calibration source points must be finite")
    delta_x = source_p2[0] - source_p1[0]
    delta_y = source_p2[1] - source_p1[1]
    source_distance = math.hypot(delta_x, delta_y)
    if not math.isfinite(source_distance) or source_distance <= 0.0:
        raise ValueError("calibration reference points must be finite and distinct")

    scale = known_distance_m / source_distance
    rotation_rad = -math.atan2(delta_y, delta_x)
    cos_theta = math.cos(rotation_rad)
    sin_theta = math.sin(rotation_rad)
    translated_x = scale * (cos_theta * source_p1[0] - sin_theta * source_p1[1])
    translated_y = scale * (sin_theta * source_p1[0] + cos_theta * source_p1[1])
    return SimilarityTransform(
        scale=scale,
        rotation_rad=rotation_rad,
        translation=(-translated_x, -translated_y),
    )


def uncalibrated_calibration(source: str | None = None) -> Calibration:
    return Calibration(status="uncalibrated", source=source)


def _transform_boundary(boundary: Boundary, transform: SimilarityTransform) -> Boundary:
    points = np.asarray(boundary.points, dtype=np.float64)
    transformed = transform.apply(points)
    return Boundary(
        points=[(float(point[0]), float(point[1])) for point in transformed],
    )


def calibrate_track_from_known_distance(
    track: Track,
    source_p1: Point2D,
    source_p2: Point2D,
    known_distance_m: float,
    *,
    source: str | None = None,
) -> Track:
    """Calibrate an uncalibrated track and return a new track in metres."""

    if track.calibration.status != "uncalibrated":
        raise ValueError("track is already calibrated")

    transform = calibrate_from_known_distance(source_p1, source_p2, known_distance_m)
    outer = _transform_boundary(track.boundaries.outer, transform)
    inner = _transform_boundary(track.boundaries.inner, transform)
    start, end = track.start_finish.line
    transformed_line = transform.apply(np.asarray([start, end], dtype=np.float64))
    transformed_forward = transform.apply_direction(track.start_finish.forward)
    start_finish = StartFinish(
        line=(
            (float(transformed_line[0, 0]), float(transformed_line[0, 1])),
            (float(transformed_line[1, 0]), float(transformed_line[1, 1])),
        ),
        forward=transformed_forward,
    )
    calibration = Calibration(
        status="calibrated",
        method="known_distance",
        scale_m_per_unit=transform.scale,
        rotation_rad=transform.rotation_rad,
        translation=transform.translation,
        reference_points=(source_p1, source_p2),
        reference_distance_m=known_distance_m,
        source=source,
    )
    return track.model_copy(
        update={
            "units": "m",
            "boundaries": TrackBoundaries(outer=outer, inner=inner),
            "start_finish": start_finish,
            "calibration": calibration,
        },
    )
