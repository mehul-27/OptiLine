"""Analytic track fixtures used by the M1.1 tests."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from optiline.core.track.calibration import uncalibrated_calibration
from optiline.core.track.geometry import as_array, resample_closed_polyline
from optiline.core.track.schema import (
    SCHEMA_VERSION,
    Boundary,
    Calibration,
    StartFinish,
    Track,
    TrackBoundaries,
)

FloatArray = npt.NDArray[np.float64]


def _points(points: FloatArray) -> list[tuple[float, float]]:
    return [(float(point[0]), float(point[1])) for point in points]


def _calibration() -> Calibration:
    return Calibration(
        status="calibrated",
        method="known_distance",
        scale_m_per_unit=1.0,
        rotation_rad=0.0,
        translation=(0.0, 0.0),
        reference_points=((0.0, 0.0), (1.0, 0.0)),
        reference_distance_m=1.0,
        source="analytic fixture",
    )


def _make_track(
    *,
    track_id: str,
    name: str,
    outer: FloatArray,
    inner: FloatArray,
    start_finish: StartFinish,
    calibrated: bool = True,
) -> Track:
    return Track(
        schema_version=SCHEMA_VERSION,
        track_id=track_id,
        name=name,
        units="m" if calibrated else "unspecified",
        boundaries=TrackBoundaries(
            outer=Boundary(points=_points(outer)),
            inner=Boundary(points=_points(inner)),
        ),
        start_finish=start_finish,
        calibration=_calibration() if calibrated else uncalibrated_calibration("analytic fixture"),
    )


def _circle(radius: float, count: int, *, clockwise: bool = False) -> FloatArray:
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    if clockwise:
        angles = -angles
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def _stadium(
    radius: float,
    straight_length: float,
    count: int,
    *,
    clockwise: bool = False,
) -> FloatArray:
    half_length = straight_length / 2.0
    cap_count = max(32, count // 8)
    straight_count = max(16, count // 4)
    right_cap = np.column_stack(
        (
            half_length + radius * np.cos(np.linspace(0.0, math.pi / 2.0, cap_count)),
            radius * np.sin(np.linspace(0.0, math.pi / 2.0, cap_count)),
        ),
    )
    top = np.column_stack(
        (
            np.linspace(half_length, -half_length, straight_count),
            np.full(straight_count, radius),
        ),
    )
    left_angles = np.linspace(math.pi / 2.0, 3.0 * math.pi / 2.0, cap_count)
    left_cap = np.column_stack(
        (
            -half_length + radius * np.cos(left_angles),
            radius * np.sin(left_angles),
        ),
    )
    bottom = np.column_stack(
        (
            np.linspace(-half_length, half_length, straight_count),
            np.full(straight_count, -radius),
        ),
    )
    lower_right_angles = np.linspace(3.0 * math.pi / 2.0, 2.0 * math.pi, cap_count)
    right_cap_lower = np.column_stack(
        (
            half_length + radius * np.cos(lower_right_angles),
            radius * np.sin(lower_right_angles),
        ),
    )
    dense = np.vstack((right_cap, top, left_cap, bottom, right_cap_lower))
    keep = np.ones(len(dense), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(dense, axis=0), axis=1) > 1e-12
    dense = dense[keep]
    if np.linalg.norm(dense[-1] - dense[0]) <= 1e-12:
        dense = dense[:-1]
    resampled = resample_closed_polyline(dense, count)
    if clockwise:
        resampled = np.vstack((resampled[0], resampled[:0:-1]))
    return resampled


def make_circle_track(
    *,
    centerline_radius: float = 50.0,
    half_width: float = 10.0,
    count: int = 360,
    calibrated: bool = True,
) -> Track:
    outer = _circle(centerline_radius + half_width, count)
    inner = _circle(centerline_radius - half_width, count, clockwise=True)
    start_finish = StartFinish(
        line=((float(inner[0, 0]), float(inner[0, 1])), (float(outer[0, 0]), float(outer[0, 1]))),
        forward=(0.0, 1.0),
    )
    return _make_track(
        track_id="circle-r50",
        name="Circle R50",
        outer=outer,
        inner=inner,
        start_finish=start_finish,
        calibrated=calibrated,
    )


def make_oval_track(
    *,
    straight_length: float = 100.0,
    centerline_radius: float = 30.0,
    half_width: float = 10.0,
    count: int = 720,
    calibrated: bool = True,
) -> Track:
    outer = _stadium(centerline_radius + half_width, straight_length, count)
    inner = _stadium(centerline_radius - half_width, straight_length, count, clockwise=True)
    start_finish = StartFinish(
        line=((float(inner[0, 0]), float(inner[0, 1])), (float(outer[0, 0]), float(outer[0, 1]))),
        forward=(0.0, 1.0),
    )
    return _make_track(
        track_id="oval-100x60",
        name="Oval 100x60",
        outer=outer,
        inner=inner,
        start_finish=start_finish,
        calibrated=calibrated,
    )


def make_uncalibrated_track() -> Track:
    return make_circle_track(calibrated=False)


def _replace_boundaries(track: Track, outer: FloatArray, inner: FloatArray) -> Track:
    return track.model_copy(
        update={
            "boundaries": TrackBoundaries(
                outer=Boundary(points=_points(outer)),
                inner=Boundary(points=_points(inner)),
            ),
        },
    )


def make_self_intersecting_track() -> Track:
    outer = np.asarray([(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)], dtype=np.float64)
    inner = np.asarray([(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)], dtype=np.float64)
    track = make_circle_track(count=64)
    return _replace_boundaries(track, outer, inner)


def make_inner_outside_track() -> Track:
    outer = _circle(10.0, 128)
    inner = _circle(20.0, 128, clockwise=True)
    track = make_circle_track(count=128)
    return _replace_boundaries(track, outer, inner)


def make_self_intersecting_start_finish_track() -> Track:
    track = make_circle_track(count=128)
    return track.model_copy(
        update={
            "start_finish": StartFinish(line=((0.0, 0.0), (60.0, 0.0)), forward=(0.0, 1.0)),
        },
    )


def make_narrow_track() -> Track:
    return make_circle_track(centerline_radius=50.0, half_width=0.2, count=256)


def make_bad_direction_track() -> Track:
    track = make_circle_track(count=128)
    return track.model_copy(
        update={
            "start_finish": StartFinish(
                line=track.start_finish.line,
                forward=(1.0, 0.0),
            ),
        },
    )


def circle_radius_points(radius: float, count: int = 360) -> FloatArray:
    return as_array([(float(x), float(y)) for x, y in _circle(radius, count)])
