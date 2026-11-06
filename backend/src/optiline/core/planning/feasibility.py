"""Independent feasibility validation for M4.1 path and speed profiles.

This module deliberately does not use the profile builder's interval-pass or
lap-time functions. It independently rechecks path chord lengths, tangents,
discrete curvature, interval acceleration/braking, lateral acceleration,
closure and lap time from the path geometry and speed samples, then separately
checks track validity and whole-segment corridor containment. Lateral checks are
pointwise at path samples; curvature between samples is outside the M4.1
piecewise-sampled path contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from shapely.geometry import LinearRing, LineString, Polygon

from optiline.core.planning.config import PlanningLimits
from optiline.core.planning.contracts import BaselinePath, SpeedProfile
from optiline.core.track.geometry import as_array
from optiline.core.track.schema import Track
from optiline.core.track.validation import normalize_track, validate_track

_LENGTH_TOL = 1e-9
_SPEED_TOL = 1e-9
_ACCEL_TOL = 1e-9
_LATERAL_TOL = 1e-9
_CURVATURE_TOL = 1e-9
_TANGENT_TOL = 1e-9
_LAP_TIME_REL_TOL = 1e-9
_LAP_TIME_ABS_TOL = 1e-9

FloatArray = npt.NDArray[np.float64]


def _independent_curvature(points: FloatArray) -> FloatArray:
    """Recompute the documented discrete curvature independently of track.geometry."""

    if len(points) < 3:
        raise ValueError("at least three points are required")
    stencil = max(1, len(points) // 64)
    stencil = max(1, min(stencil, len(points) // 2))
    previous = np.roll(points, stencil, axis=0)
    following = np.roll(points, -stencil, axis=0)
    a = np.linalg.norm(points - previous, axis=1)
    b = np.linalg.norm(following - points, axis=1)
    c = np.linalg.norm(following - previous, axis=1)
    cross = (points[:, 0] - previous[:, 0]) * (following[:, 1] - previous[:, 1]) - (
        points[:, 1] - previous[:, 1]
    ) * (following[:, 0] - previous[:, 0])
    denominator = a * b * c
    curvature: FloatArray = np.zeros_like(denominator)
    np.divide(2.0 * cross, denominator, out=curvature, where=denominator > 1e-12)
    return curvature


@dataclass(frozen=True)
class FeasibilityIssue:
    """One independently detected feasibility issue."""

    code: str
    message: str
    sample_index: int | None = None
    severity: Literal["error", "warning"] = "error"


@dataclass(frozen=True)
class FeasibilityReport:
    """Structured independent validation result."""

    feasible: bool
    issues: tuple[FeasibilityIssue, ...]
    recomputed_lap_time_s: float | None
    max_lateral_acceleration_mps2: float | None
    max_longitudinal_acceleration_mps2: float | None
    max_braking_deceleration_mps2: float | None


def _add_issue(
    issues: list[FeasibilityIssue],
    code: str,
    message: str,
    *,
    sample_index: int | None = None,
) -> None:
    issues.append(
        FeasibilityIssue(
            code=code,
            message=message,
            sample_index=sample_index,
        ),
    )


def _check_track_and_corridor(
    track: Track,
    path: BaselinePath,
    issues: list[FeasibilityIssue],
) -> None:
    try:
        track_report = validate_track(track)
    except Exception as exc:
        _add_issue(issues, "TRACK_VALIDATION_FAILED", str(exc))
        return

    if not track_report.simulation_ready:
        codes = sorted({issue.code for issue in track_report.issues if issue.severity == "error"})
        detail = ", ".join(codes) if codes else "track is not simulation-ready"
        _add_issue(issues, "TRACK_NOT_SIMULATION_READY", detail)

    try:
        normalized, _ = normalize_track(track)
        outer = as_array(normalized.boundaries.outer.points)
        inner = as_array(normalized.boundaries.inner.points)
        outer_ring = LinearRing(np.vstack((outer, outer[0])))
        inner_ring = LinearRing(np.vstack((inner, inner[0])))
        outer_polygon = Polygon(outer_ring)
        inner_polygon = Polygon(inner_ring)
        corridor = Polygon(outer_ring, [inner_ring])
        if not outer_polygon.is_valid or not inner_polygon.is_valid or not corridor.is_valid:
            _add_issue(issues, "TRACK_CORRIDOR_INVALID", "track corridor or boundaries are invalid")
            return

        bounds = outer_polygon.bounds
        extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 1.0)
        tolerance = max(1e-9, 1e-6 * extent)
        path_line = LineString([(sample.x_m, sample.y_m) for sample in path.samples])
        if not path_line.is_valid or path_line.length <= tolerance:
            _add_issue(issues, "PATH_GEOMETRY_INVALID", "baseline path line is invalid")
            return
        if not corridor.buffer(tolerance).covers(path_line):
            _add_issue(
                issues,
                "PATH_OUTSIDE_CORRIDOR",
                "baseline path is not wholly contained in the drivable corridor",
            )
    except Exception as exc:
        _add_issue(issues, "TRACK_CORRIDOR_CHECK_FAILED", str(exc))


def _check_path_contract(
    track: Track,
    path: BaselinePath,
    issues: list[FeasibilityIssue],
) -> None:
    if path.source_track_id != track.track_id:
        _add_issue(issues, "PATH_TRACK_MISMATCH", "path source_track_id does not match track")
    if not path.closed:
        _add_issue(issues, "PATH_NOT_CLOSED", "baseline path must be closed")
    if path.unique_sample_count < 3:
        _add_issue(issues, "PATH_SAMPLE_COUNT_INVALID", "path needs at least three unique samples")
    if len(path.samples) != path.unique_sample_count + 1:
        _add_issue(issues, "PATH_SAMPLE_COUNT_INVALID", "path must contain repeated final endpoint")
    if not path.samples:
        _add_issue(issues, "PATH_EMPTY", "path has no samples")
        return
    if abs(path.samples[0].s_m) > _LENGTH_TOL:
        _add_issue(issues, "PATH_START_INVALID", "path must start at s=0")
    if abs(path.samples[-1].s_m - path.length_m) > _LENGTH_TOL:
        _add_issue(issues, "PATH_LENGTH_MISMATCH", "path final sample does not match length_m")
    first = path.samples[0]
    last = path.samples[-1]
    if abs(first.x_m - last.x_m) > _LENGTH_TOL or abs(first.y_m - last.y_m) > _LENGTH_TOL:
        _add_issue(issues, "PATH_NOT_CLOSED", "path start and end positions do not match")

    geometric_length = 0.0
    for index in range(len(path.samples) - 1):
        first_sample = path.samples[index]
        second_sample = path.samples[index + 1]
        chord = math.hypot(
            second_sample.x_m - first_sample.x_m,
            second_sample.y_m - first_sample.y_m,
        )
        arc_interval = second_sample.s_m - first_sample.s_m
        if abs(chord - arc_interval) > _LENGTH_TOL:
            _add_issue(
                issues,
                "PATH_ARC_LENGTH_MISMATCH",
                "path arc-length interval does not match geometric chord",
                sample_index=index,
            )
        geometric_length += chord
    if abs(geometric_length - path.length_m) > _LENGTH_TOL:
        _add_issue(
            issues,
            "PATH_LENGTH_MISMATCH",
            "path length does not match summed geometric chord length",
        )

    try:
        positions = np.asarray([(sample.x_m, sample.y_m) for sample in path.samples[:-1]])
        recomputed_curvature = _independent_curvature(positions)
        recomputed_curvature = np.concatenate((recomputed_curvature, recomputed_curvature[:1]))
        for index, sample in enumerate(path.samples):
            if abs(sample.curvature_1_per_m - float(recomputed_curvature[index])) > _CURVATURE_TOL:
                _add_issue(
                    issues,
                    "PATH_CURVATURE_MISMATCH",
                    "stored path curvature does not match independent geometric curvature",
                    sample_index=index,
                )
    except Exception as exc:
        _add_issue(issues, "PATH_CURVATURE_CHECK_FAILED", str(exc))

    for index in range(len(path.samples) - 1):
        first_sample = path.samples[index]
        second_sample = path.samples[index + 1]
        dx = second_sample.x_m - first_sample.x_m
        dy = second_sample.y_m - first_sample.y_m
        chord = math.hypot(dx, dy)
        if chord <= _LENGTH_TOL:
            continue
        alignment = (first_sample.tangent_x * dx + first_sample.tangent_y * dy) / chord
        if abs(alignment - 1.0) > _TANGENT_TOL:
            _add_issue(
                issues,
                "PATH_TANGENT_MISMATCH",
                "path tangent does not align with the outgoing path segment",
                sample_index=index,
            )
    if len(path.samples) >= 2 and (
        abs(path.samples[-1].tangent_x - path.samples[0].tangent_x) > _TANGENT_TOL
        or abs(path.samples[-1].tangent_y - path.samples[0].tangent_y) > _TANGENT_TOL
    ):
        _add_issue(
            issues,
            "PATH_TANGENT_CLOSURE_MISMATCH",
            "path final tangent does not match path start tangent",
            sample_index=len(path.samples) - 1,
        )

    previous = -math.inf
    for index, sample in enumerate(path.samples):
        values = (
            sample.x_m,
            sample.y_m,
            sample.s_m,
            sample.tangent_x,
            sample.tangent_y,
            sample.curvature_1_per_m,
        )
        if not all(math.isfinite(value) for value in values):
            _add_issue(
                issues, "PATH_NON_FINITE", "path contains a non-finite value", sample_index=index
            )
            continue
        if sample.s_m <= previous:
            _add_issue(
                issues, "PATH_INTERVAL_INVALID", "path arc length must increase", sample_index=index
            )
        previous = sample.s_m
        tangent_norm = math.hypot(sample.tangent_x, sample.tangent_y)
        if abs(tangent_norm - 1.0) > _LENGTH_TOL:
            _add_issue(
                issues,
                "PATH_TANGENT_INVALID",
                "path tangent is not unit length",
                sample_index=index,
            )
        if index > 0:
            interval = sample.s_m - path.samples[index - 1].s_m
            if not math.isfinite(interval) or interval <= 0.0:
                _add_issue(
                    issues,
                    "PATH_INTERVAL_INVALID",
                    "path contains a non-positive interval",
                    sample_index=index,
                )


def _check_profile_contract(
    path: BaselinePath,
    profile: SpeedProfile,
    limits: PlanningLimits,
    issues: list[FeasibilityIssue],
) -> None:
    if profile.source_track_id != path.source_track_id:
        _add_issue(issues, "PROFILE_TRACK_MISMATCH", "profile and path source ids differ")
    if not profile.closed:
        _add_issue(issues, "PROFILE_NOT_CLOSED", "speed profile must be closed")
    if profile.unique_sample_count != path.unique_sample_count:
        _add_issue(issues, "PROFILE_SAMPLE_COUNT_INVALID", "profile and path sample counts differ")
    if len(profile.samples) != path.unique_sample_count + 1:
        _add_issue(issues, "PROFILE_SAMPLE_COUNT_INVALID", "profile must contain repeated endpoint")
    if len(profile.interval_longitudinal_accelerations_mps2) != path.unique_sample_count:
        _add_issue(
            issues,
            "PROFILE_INTERVAL_COUNT_INVALID",
            "profile must contain one interval acceleration per interval",
        )
    if abs(profile.path_length_m - path.length_m) > _LENGTH_TOL:
        _add_issue(issues, "PROFILE_LENGTH_MISMATCH", "profile path length differs from path")
    if profile.limits != limits:
        _add_issue(issues, "PROFILE_LIMITS_MISMATCH", "profile limits differ from validator limits")

    if not profile.samples:
        _add_issue(issues, "PROFILE_EMPTY", "profile has no samples")
        return
    if abs(profile.samples[0].s_m) > _LENGTH_TOL:
        _add_issue(issues, "PROFILE_START_INVALID", "profile must start at s=0")
    if abs(profile.samples[-1].s_m - profile.path_length_m) > _LENGTH_TOL:
        _add_issue(issues, "PROFILE_LENGTH_MISMATCH", "profile final sample does not match length")
    if abs(profile.samples[0].speed_mps - profile.samples[-1].speed_mps) > _SPEED_TOL:
        _add_issue(issues, "PROFILE_CLOSURE_SPEED_MISMATCH", "profile start and end speeds differ")

    for index, sample in enumerate(profile.samples):
        values = (
            sample.s_m,
            sample.speed_mps,
            sample.curvature_1_per_m,
            sample.lateral_acceleration_mps2,
        )
        if not all(math.isfinite(value) for value in values):
            _add_issue(
                issues,
                "PROFILE_NON_FINITE",
                "profile contains a non-finite value",
                sample_index=index,
            )
            continue
        if sample.speed_mps <= 0.0:
            _add_issue(
                issues,
                "PROFILE_NON_POSITIVE_SPEED",
                "profile speed must be positive",
                sample_index=index,
            )
        if sample.speed_mps > limits.max_speed_mps + _SPEED_TOL:
            _add_issue(
                issues,
                "PROFILE_MAX_SPEED_EXCEEDED",
                "profile exceeds maximum speed",
                sample_index=index,
            )
        if index < len(path.samples):
            if abs(sample.s_m - path.samples[index].s_m) > _LENGTH_TOL:
                _add_issue(
                    issues,
                    "PROFILE_ARC_LENGTH_MISMATCH",
                    "profile and path arc lengths differ",
                    sample_index=index,
                )
            if abs(sample.curvature_1_per_m - path.samples[index].curvature_1_per_m) > _LENGTH_TOL:
                _add_issue(
                    issues,
                    "PROFILE_CURVATURE_MISMATCH",
                    "profile and path curvature differ",
                    sample_index=index,
                )
        expected_lateral = sample.speed_mps * sample.speed_mps * abs(sample.curvature_1_per_m)
        if abs(sample.lateral_acceleration_mps2 - expected_lateral) > _LATERAL_TOL:
            _add_issue(
                issues,
                "PROFILE_LATERAL_MISMATCH",
                "stored lateral acceleration does not match speed and curvature",
                sample_index=index,
            )


def validate_speed_profile(
    track: Track,
    path: BaselinePath,
    profile: SpeedProfile,
    limits: PlanningLimits,
) -> FeasibilityReport:
    """Independently validate a closed path and its flying-lap speed profile."""

    issues: list[FeasibilityIssue] = []
    _check_track_and_corridor(track, path, issues)
    _check_path_contract(track, path, issues)
    _check_profile_contract(path, profile, limits, issues)

    max_lateral: float | None = None
    max_acceleration: float | None = None
    max_braking: float | None = None
    recomputed_lap_time: float | None = None

    n = path.unique_sample_count
    if len(path.samples) == n + 1 and len(profile.samples) == n + 1:
        lateral_values: list[float] = []
        accelerations: list[float] = []
        braking_values: list[float] = []
        interval_times: list[float] = []
        intervals_ok = True

        for index in range(n):
            first_path = path.samples[index]
            second_path = path.samples[index + 1]
            first_speed = profile.samples[index].speed_mps
            second_speed = profile.samples[index + 1].speed_mps
            ds = second_path.s_m - first_path.s_m
            if not math.isfinite(ds) or ds <= 0.0:
                _add_issue(
                    issues,
                    "PROFILE_INTERVAL_INVALID",
                    "profile interval is non-positive",
                    sample_index=index,
                )
                intervals_ok = False
                continue
            if not (math.isfinite(first_speed) and math.isfinite(second_speed)):
                _add_issue(
                    issues, "PROFILE_NON_FINITE", "profile speed is non-finite", sample_index=index
                )
                intervals_ok = False
                continue
            if first_speed <= 0.0 or second_speed <= 0.0:
                intervals_ok = False
                continue

            acceleration = (second_speed * second_speed - first_speed * first_speed) / (2.0 * ds)
            accelerations.append(acceleration)
            if acceleration < 0.0:
                braking_values.append(-acceleration)
            if acceleration > limits.max_longitudinal_acceleration_mps2 + _ACCEL_TOL:
                _add_issue(
                    issues,
                    "LONGITUDINAL_ACCELERATION_EXCEEDED",
                    "profile exceeds maximum longitudinal acceleration",
                    sample_index=index,
                )
            if acceleration < -limits.max_braking_deceleration_mps2 - _ACCEL_TOL:
                _add_issue(
                    issues,
                    "BRAKING_EXCEEDED",
                    "profile exceeds maximum braking deceleration",
                    sample_index=index,
                )
            if index < len(profile.interval_longitudinal_accelerations_mps2):
                stored = profile.interval_longitudinal_accelerations_mps2[index]
                if abs(stored - acceleration) > _ACCEL_TOL:
                    _add_issue(
                        issues,
                        "PROFILE_ACCELERATION_MISMATCH",
                        "stored interval acceleration does not match speeds",
                        sample_index=index,
                    )
            interval_times.append(2.0 * ds / (first_speed + second_speed))

        for index, sample in enumerate(profile.samples):
            if not math.isfinite(sample.speed_mps) or not math.isfinite(sample.curvature_1_per_m):
                continue
            lateral = sample.speed_mps * sample.speed_mps * abs(sample.curvature_1_per_m)
            lateral_values.append(lateral)
            if lateral > limits.max_lateral_acceleration_mps2 + _LATERAL_TOL:
                _add_issue(
                    issues,
                    "LATERAL_ACCELERATION_EXCEEDED",
                    "profile exceeds maximum lateral acceleration",
                    sample_index=index,
                )

        if accelerations:
            max_lateral = max(lateral_values) if lateral_values else None
            max_acceleration = max(accelerations)
            max_braking = max(braking_values) if braking_values else 0.0
        if intervals_ok and len(interval_times) == n:
            lap_time = math.fsum(interval_times)
            if math.isfinite(lap_time) and lap_time > 0.0:
                recomputed_lap_time = lap_time
                tolerance = max(_LAP_TIME_ABS_TOL, _LAP_TIME_REL_TOL * abs(lap_time))
                if abs(profile.lap_time_s - lap_time) > tolerance:
                    _add_issue(
                        issues,
                        "LAP_TIME_MISMATCH",
                        "stored lap time does not match independently recomputed lap time",
                    )

    feasible = not any(issue.severity == "error" for issue in issues)
    return FeasibilityReport(
        feasible=feasible,
        issues=tuple(issues),
        recomputed_lap_time_s=recomputed_lap_time,
        max_lateral_acceleration_mps2=max_lateral,
        max_longitudinal_acceleration_mps2=max_acceleration,
        max_braking_deceleration_mps2=max_braking,
    )
