"""Track validation using Shapely plus OptiLine-specific corridor rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from shapely.geometry import LinearRing, LineString, Point, Polygon
from shapely.validation import explain_validity

from optiline.core.track.geometry import (
    TrackSummary,
    as_array,
    ensure_ccw,
    is_ccw,
    summarize_track,
)
from optiline.core.track.schema import Boundary, Track, TrackBoundaries

BoundaryName = Literal["outer", "inner", "start_finish", "calibration", "schema", "centerline"]
Severity = Literal["error", "warning"]

# The derived centerline is sampled from the outer boundary. A segment much
# longer than the expected uniform spacing indicates that nearest-point
# pairing jumped across a concavity instead of following the corridor.
CENTERLINE_CONTINUITY_RATIO_LIMIT = 5.0


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Severity
    message: str
    boundary: BoundaryName | None = None
    point_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class TrackValidationReport:
    """Structured track validation result.

    ``geometry_valid`` covers boundary, corridor and start/finish geometry. It
    excludes calibration errors and derived-centerline continuity errors so a
    boundary-valid track can still be edited while remaining not
    simulation-ready.
    """

    is_valid: bool
    geometry_valid: bool
    simulation_ready: bool
    issues: tuple[ValidationIssue, ...]
    summary: TrackSummary | None


def _as_boundary(points: np.ndarray) -> Boundary:
    return Boundary(
        points=[(float(point[0]), float(point[1])) for point in points],
    )


def normalize_track(track: Track) -> tuple[Track, list[ValidationIssue]]:
    """Normalize boundary orientation to outer CCW and inner CW."""

    issues: list[ValidationIssue] = []
    outer_points, outer_reversed = ensure_ccw(as_array(track.boundaries.outer.points))
    inner_points = as_array(track.boundaries.inner.points)
    inner_reversed = False
    if is_ccw(inner_points):
        inner_points = np.ascontiguousarray(inner_points[::-1])
        inner_reversed = True

    if outer_reversed:
        issues.append(
            ValidationIssue(
                code="BOUNDARY_ORIENTATION_NORMALIZED",
                severity="warning",
                message="outer boundary was reversed to counter-clockwise order",
                boundary="outer",
            ),
        )
    if inner_reversed:
        issues.append(
            ValidationIssue(
                code="BOUNDARY_ORIENTATION_NORMALIZED",
                severity="warning",
                message="inner boundary was reversed to clockwise order",
                boundary="inner",
            ),
        )

    if not outer_reversed and not inner_reversed:
        return track, issues

    return track.model_copy(
        update={
            "boundaries": TrackBoundaries(
                outer=_as_boundary(outer_points),
                inner=_as_boundary(inner_points),
            ),
        },
    ), issues


def validate_track(
    track: Track,
    *,
    count: int = 4096,
    min_width_m: float = 1.0,
) -> TrackValidationReport:
    """Validate a track and return structured issues plus derived geometry."""

    normalized, issues = normalize_track(track)
    outer = as_array(normalized.boundaries.outer.points)
    inner = as_array(normalized.boundaries.inner.points)
    outer_coords = np.vstack((outer, outer[0]))
    inner_coords = np.vstack((inner, inner[0]))

    outer_ring = LinearRing(outer_coords)
    inner_ring = LinearRing(inner_coords)
    outer_line = LineString(outer_coords)
    inner_line = LineString(inner_coords)
    outer_polygon = Polygon(outer_ring)
    inner_polygon = Polygon(inner_ring)
    corridor = Polygon(outer_ring, [inner_ring])

    for name, polygon in (("outer", outer_polygon), ("inner", inner_polygon)):
        if not polygon.is_valid:
            boundary_name: Literal["outer", "inner"] = "outer" if name == "outer" else "inner"
            issues.append(
                ValidationIssue(
                    code="BOUNDARY_SELF_INTERSECTION",
                    severity="error",
                    message=f"{name} boundary is invalid: {explain_validity(polygon)}",
                    boundary=boundary_name,
                ),
            )

    if outer_line.intersects(inner_line):
        issues.append(
            ValidationIssue(
                code="BOUNDARIES_INTERSECT",
                severity="error",
                message="outer and inner boundaries intersect or touch",
                boundary="outer",
            ),
        )

    try:
        contains_inner = outer_polygon.contains(inner_polygon)
    except Exception:
        contains_inner = False
    if not contains_inner:
        issues.append(
            ValidationIssue(
                code="INNER_NOT_INSIDE_OUTER",
                severity="error",
                message="inner boundary must lie strictly inside the outer boundary",
                boundary="inner",
            ),
        )

    if not corridor.is_valid or corridor.area <= 0.0:
        issues.append(
            ValidationIssue(
                code="CORRIDOR_INVALID",
                severity="error",
                message=f"track corridor is invalid: {explain_validity(corridor)}",
                boundary="outer",
            ),
        )

    bounds = outer_polygon.bounds
    extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 1.0)
    tolerance = max(1e-9, 1e-6 * extent)
    first, second = normalized.start_finish.line
    first_to_outer = Point(first).distance(outer_line)
    first_to_inner = Point(first).distance(inner_line)
    second_to_outer = Point(second).distance(outer_line)
    second_to_inner = Point(second).distance(inner_line)

    if min(first_to_outer, first_to_inner) > tolerance:
        issues.append(
            ValidationIssue(
                code="START_FINISH_ENDPOINT_NOT_ON_BOUNDARY",
                severity="error",
                message="first start/finish endpoint is not on a boundary",
                boundary="start_finish",
                point_indices=(0,),
            ),
        )
    if min(second_to_outer, second_to_inner) > tolerance:
        issues.append(
            ValidationIssue(
                code="START_FINISH_ENDPOINT_NOT_ON_BOUNDARY",
                severity="error",
                message="second start/finish endpoint is not on a boundary",
                boundary="start_finish",
                point_indices=(1,),
            ),
        )

    first_closest_to_outer = first_to_outer <= first_to_inner
    second_closest_to_outer = second_to_outer <= second_to_inner
    if first_closest_to_outer == second_closest_to_outer:
        issues.append(
            ValidationIssue(
                code="START_FINISH_ENDPOINT_NOT_ON_BOUNDARY",
                severity="error",
                message="start/finish endpoints must connect the outer and inner boundaries",
                boundary="start_finish",
            ),
        )

    start_finish_line = LineString([first, second])
    try:
        spans_corridor = corridor.buffer(tolerance).covers(start_finish_line)
    except Exception:
        spans_corridor = False
    if not spans_corridor or start_finish_line.length <= tolerance:
        issues.append(
            ValidationIssue(
                code="START_FINISH_DOES_NOT_SPAN_CORRIDOR",
                severity="error",
                message="start/finish line must span the drivable corridor",
                boundary="start_finish",
            ),
        )

    line_vector = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    line_length = float(np.linalg.norm(line_vector))
    if line_length > tolerance:
        line_unit = line_vector / line_length
        forward = np.asarray(normalized.start_finish.forward, dtype=np.float64)
        if abs(float(np.dot(forward, line_unit))) > 0.1:
            issues.append(
                ValidationIssue(
                    code="START_FINISH_DIRECTION_INVALID",
                    severity="error",
                    message="start/finish forward vector must be across the track, not along it",
                    boundary="start_finish",
                ),
            )

    summary = None
    try:
        summary = summarize_track(normalized, count=count)
    except Exception as exc:
        issues.append(
            ValidationIssue(
                code="CORRIDOR_INVALID",
                severity="error",
                message=f"could not derive track geometry: {exc}",
                boundary="outer",
            ),
        )

    if summary is not None:
        if summary.min_width <= 0.0:
            issues.append(
                ValidationIssue(
                    code="CORRIDOR_INVALID",
                    severity="error",
                    message="track corridor width must be positive",
                    boundary="outer",
                ),
            )
        if (
            normalized.calibration.status == "calibrated"
            and summary.min_width_m is not None
            and summary.min_width_m < min_width_m
        ):
            issues.append(
                ValidationIssue(
                    code="CORRIDOR_TOO_NARROW",
                    severity="error",
                    message=(
                        f"minimum corridor width {summary.min_width_m:.3f} m is below "
                        f"{min_width_m:.3f} m"
                    ),
                    boundary="outer",
                ),
            )

        segment_lengths = np.linalg.norm(
            np.roll(summary.centerline, -1, axis=0) - summary.centerline,
            axis=1,
        )
        expected_spacing = summary.length_world_units / len(summary.centerline)
        max_segment = float(np.max(segment_lengths))
        if expected_spacing > 0.0 and max_segment > (
            CENTERLINE_CONTINUITY_RATIO_LIMIT * expected_spacing
        ):
            issues.append(
                ValidationIssue(
                    code="CENTERLINE_DISCONTINUITY",
                    severity="error",
                    message=(
                        "derived centerline has a discontinuity: maximum segment "
                        f"{max_segment:.3f} exceeds {CENTERLINE_CONTINUITY_RATIO_LIMIT:.0f}x "
                        f"the expected spacing {expected_spacing:.3f}"
                    ),
                    boundary="centerline",
                ),
            )

    calibrated = normalized.calibration.status == "calibrated" and normalized.units == "m"
    if not calibrated:
        issues.append(
            ValidationIssue(
                code="CALIBRATION_MISSING",
                severity="error",
                message="track must be calibrated in metres before simulation",
                boundary="calibration",
            ),
        )

    is_valid = not any(issue.severity == "error" for issue in issues)
    geometry_valid = not any(
        issue.severity == "error" and issue.boundary not in ("calibration", "schema", "centerline")
        for issue in issues
    )
    simulation_ready = (
        is_valid
        and calibrated
        and summary is not None
        and summary.min_width_m is not None
        and summary.min_width_m >= min_width_m
    )
    return TrackValidationReport(
        is_valid=is_valid,
        geometry_valid=geometry_valid,
        simulation_ready=simulation_ready,
        issues=tuple(issues),
        summary=summary,
    )
