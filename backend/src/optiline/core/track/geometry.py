"""Planar geometry used by the track domain."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from optiline.core.track.schema import Point2D, Track

FloatArray = npt.NDArray[np.float64]
_EPSILON = 1e-12


@dataclass(frozen=True)
class CenterlineResult:
    """Sampled centerline and the paired boundary samples used to derive it."""

    centerline: FloatArray
    outer_samples: FloatArray
    inner_samples: FloatArray
    widths: FloatArray
    outer_start: Point2D
    inner_start: Point2D


@dataclass(frozen=True)
class TrackSummary:
    """Derived geometry for a track; never persisted as authoritative data."""

    length_world_units: float
    lap_length_m: float | None
    min_width: float
    max_width: float
    min_width_m: float | None
    centerline: FloatArray
    curvature: FloatArray
    widths: FloatArray


def as_array(points: Sequence[Point2D]) -> FloatArray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    return array


def signed_area(points: FloatArray) -> float:
    if len(points) < 3:
        raise ValueError("at least three points are required")
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def is_ccw(points: FloatArray) -> bool:
    return signed_area(points) > 0.0


def ensure_ccw(points: FloatArray) -> tuple[FloatArray, bool]:
    if is_ccw(points):
        return points, False
    return np.ascontiguousarray(points[::-1]), True


def closed_polyline_length(points: FloatArray) -> float:
    if len(points) < 2:
        raise ValueError("at least two points are required")
    return float(np.sum(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)))


def _segment_lengths(points: FloatArray) -> FloatArray:
    return np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)


def project_point_to_closed_polyline(
    points: FloatArray,
    target: Point2D,
) -> tuple[FloatArray, int, float, float]:
    """Project a point onto the closest segment of a closed polyline."""

    if len(points) < 3:
        raise ValueError("at least three points are required")
    target_array = np.asarray(target, dtype=np.float64)
    segments = np.roll(points, -1, axis=0) - points
    segment_length_squared = np.einsum("ij,ij->i", segments, segments)
    safe_lengths = np.where(segment_length_squared > _EPSILON, segment_length_squared, np.inf)
    offsets = target_array - points
    parameters = np.clip(
        np.einsum("ij,ij->i", offsets, segments) / safe_lengths,
        0.0,
        1.0,
    )
    projections = points + parameters[:, None] * segments
    distances = np.linalg.norm(projections - target_array, axis=1)
    index = int(np.argmin(distances))
    return projections[index], index, float(parameters[index]), float(distances[index])


def _forward_order(points: FloatArray, start_point: Point2D) -> FloatArray:
    projected, segment_index, parameter, _ = project_point_to_closed_polyline(points, start_point)
    vertex_count = len(points)
    if parameter <= _EPSILON:
        return np.roll(points, -segment_index, axis=0)
    if parameter >= 1.0 - _EPSILON:
        return np.roll(points, -((segment_index + 1) % vertex_count), axis=0)
    return np.vstack(
        (
            projected[None, :],
            np.roll(points, -((segment_index + 1) % vertex_count), axis=0),
        ),
    )


def initial_tangent(ordered: FloatArray) -> FloatArray:
    for index in range(len(ordered)):
        next_index = (index + 1) % len(ordered)
        delta = ordered[next_index] - ordered[index]
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > _EPSILON:
            tangent: FloatArray = np.asarray(delta / delta_norm, dtype=np.float64)
            return tangent
    raise ValueError("polyline has no non-zero segment")


def resample_closed_polyline(
    points: FloatArray,
    count: int,
    *,
    start_point: Point2D | None = None,
    direction: int = 1,
) -> FloatArray:
    """Resample a closed polyline by arc length from an optional start point."""

    if count < 3:
        raise ValueError("resample count must be at least three")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")

    ordered = points.copy() if start_point is None else _forward_order(points, start_point)
    if direction < 0:
        ordered = np.vstack((ordered[0], ordered[:0:-1]))

    segment_vectors = np.roll(ordered, -1, axis=0) - ordered
    lengths = np.linalg.norm(segment_vectors, axis=1)
    if np.any(lengths <= _EPSILON):
        raise ValueError("polyline contains duplicate consecutive points")
    cumulative = np.concatenate((np.zeros(1), np.cumsum(lengths)))
    targets = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    segment_indices = np.searchsorted(cumulative, targets, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, len(ordered) - 1)
    local_parameters = (targets - cumulative[segment_indices]) / lengths[segment_indices]
    return ordered[segment_indices] + local_parameters[:, None] * segment_vectors[segment_indices]


def _start_points(track: Track) -> tuple[Point2D, Point2D]:
    outer = as_array(track.boundaries.outer.points)
    inner = as_array(track.boundaries.inner.points)
    first, second = track.start_finish.line
    _, _, _, first_to_outer = project_point_to_closed_polyline(outer, first)
    _, _, _, first_to_inner = project_point_to_closed_polyline(inner, first)
    _, _, _, second_to_outer = project_point_to_closed_polyline(outer, second)
    _, _, _, second_to_inner = project_point_to_closed_polyline(inner, second)

    if first_to_outer + second_to_inner <= first_to_inner + second_to_outer:
        return first, second
    return second, first


def derive_centerline(track: Track, count: int = 4096) -> CenterlineResult:
    """Derive a baseline centerline from boundary samples.

    The outer boundary is sampled from the start/finish in the travel
    direction. Each sample is paired with its closest point on the inner
    boundary. This avoids the arc-length parameterization mismatch that
    appears when inner and outer boundaries have different cap proportions.
    The result is a documented boundary-midpoint baseline, not a racing line
    and not a general medial-axis solution.
    """

    outer = as_array(track.boundaries.outer.points)
    inner = as_array(track.boundaries.inner.points)
    outer_start, inner_start = _start_points(track)
    forward = np.asarray(track.start_finish.forward, dtype=np.float64)

    outer_ordered = _forward_order(outer, outer_start)
    outer_direction = 1 if float(np.dot(initial_tangent(outer_ordered), forward)) >= 0.0 else -1
    outer_samples = resample_closed_polyline(
        outer,
        count,
        start_point=outer_start,
        direction=outer_direction,
    )
    inner_samples = np.vstack(
        [project_point_to_closed_polyline(inner, point)[0] for point in outer_samples],
    )

    centerline = 0.5 * (outer_samples + inner_samples)
    widths = np.linalg.norm(outer_samples - inner_samples, axis=1)
    return CenterlineResult(
        centerline=centerline,
        outer_samples=outer_samples,
        inner_samples=inner_samples,
        widths=widths,
        outer_start=outer_start,
        inner_start=inner_start,
    )


def compute_curvature(
    centerline: FloatArray,
    stencil: int | None = None,
) -> FloatArray:
    """Compute signed discrete curvature at each centerline sample.

    The default stencil spans roughly 1/64 of the centerline. A wider stencil
    suppresses polygon-vertex noise from the boundary-midpoint baseline; near
    sharp corners the estimate is therefore a local average, not a pointwise
    geometric curvature.
    """

    if len(centerline) < 3:
        raise ValueError("at least three centerline points are required")
    if stencil is None:
        stencil = max(1, len(centerline) // 64)
    stencil = max(1, min(stencil, len(centerline) // 2))
    previous = np.roll(centerline, stencil, axis=0)
    following = np.roll(centerline, -stencil, axis=0)
    a = np.linalg.norm(centerline - previous, axis=1)
    b = np.linalg.norm(following - centerline, axis=1)
    c = np.linalg.norm(following - previous, axis=1)
    cross = (centerline[:, 0] - previous[:, 0]) * (following[:, 1] - previous[:, 1]) - (
        centerline[:, 1] - previous[:, 1]
    ) * (following[:, 0] - previous[:, 0])
    denominator = a * b * c
    curvature = np.zeros_like(denominator)
    np.divide(2.0 * cross, denominator, out=curvature, where=denominator > _EPSILON)
    return curvature


def summarize_track(track: Track, count: int = 4096) -> TrackSummary:
    result = derive_centerline(track, count=count)
    length_world_units = closed_polyline_length(result.centerline)
    calibrated = track.calibration.status == "calibrated" and track.units == "m"
    min_width = float(np.min(result.widths))
    max_width = float(np.max(result.widths))
    return TrackSummary(
        length_world_units=length_world_units,
        lap_length_m=length_world_units if calibrated else None,
        min_width=min_width,
        max_width=max_width,
        min_width_m=min_width if calibrated else None,
        centerline=result.centerline,
        curvature=compute_curvature(result.centerline),
        widths=result.widths,
    )
