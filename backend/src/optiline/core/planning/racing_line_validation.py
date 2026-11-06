"""Independent M4.2 validation layered over the M4.1 path/speed validator."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from shapely.geometry import LinearRing, LineString

from optiline.core.planning.config import PlanningLimits
from optiline.core.planning.contracts import BaselinePath, SpeedProfile
from optiline.core.planning.feasibility import (
    FeasibilityIssue,
    FeasibilityReport,
    validate_speed_profile,
)
from optiline.core.planning.racing_line import (
    BoundaryOffsetError,
    RacingLinePath,
    baseline_fingerprint,
    build_offset_bounds,
)
from optiline.core.track.geometry import as_array
from optiline.core.track.schema import Track
from optiline.core.track.validation import normalize_track

_LENGTH_TOL = 1e-9
_OFFSET_TOL = 1e-7
_MARGIN_TOL = 1e-7
_LOOP_TOL = 1e-9


@dataclass(frozen=True)
class RacingLineValidationReport:
    """M4.1 validation plus racing-line-specific independent checks."""

    feasible: bool
    issues: tuple[FeasibilityIssue, ...]
    m4_report: FeasibilityReport
    lap_time_s: float | None


def _add_issue(
    issues: list[FeasibilityIssue],
    code: str,
    message: str,
    *,
    sample_index: int | None = None,
) -> None:
    issues.append(FeasibilityIssue(code=code, message=message, sample_index=sample_index))


def _periodic_spline(
    control_offsets: tuple[float, ...],
    length_m: float,
) -> CubicSpline:
    control = np.asarray(control_offsets, dtype=np.float64)
    knots = np.linspace(0.0, length_m, len(control) + 1)
    values = np.concatenate((control, control[:1]))
    return CubicSpline(knots, values, bc_type="periodic", extrapolate="periodic")


def _check_offsets(
    track: Track,
    baseline: BaselinePath,
    path: RacingLinePath,
    issues: list[FeasibilityIssue],
) -> None:
    if path.source_track_id != track.track_id:
        _add_issue(issues, "RACING_LINE_TRACK_MISMATCH", "racing-line track id differs from track")
    if path.source_track_id != baseline.source_track_id:
        _add_issue(
            issues,
            "RACING_LINE_BASELINE_TRACK_MISMATCH",
            "racing-line track id differs from the reference baseline",
        )
    expected_reference = baseline_fingerprint(baseline)
    if path.reference_baseline_id != expected_reference:
        _add_issue(
            issues,
            "RACING_LINE_BASELINE_MISMATCH",
            "racing-line reference baseline identity differs from the supplied baseline",
        )
    if path.unique_sample_count != baseline.unique_sample_count:
        _add_issue(
            issues,
            "RACING_LINE_SAMPLE_COUNT_MISMATCH",
            "racing-line and baseline sample counts differ",
        )
        return
    if not (math.isfinite(path.boundary_margin_m) and path.boundary_margin_m > 0.0):
        _add_issue(issues, "BOUNDARY_MARGIN_INVALID", "boundary margin must be positive and finite")
    if abs(path.offset_samples_m[0] - path.offset_samples_m[-1]) > _LENGTH_TOL:
        _add_issue(
            issues,
            "RACING_LINE_OFFSET_CLOSURE_MISMATCH",
            "racing-line start and end offsets differ",
        )
    if len(path.control_offsets_m) < 4:
        _add_issue(issues, "SPLINE_CONTROL_COUNT_INVALID", "fewer than four spline controls")
        return

    try:
        spline = _periodic_spline(path.control_offsets_m, baseline.length_m)
        reference_s = np.asarray(
            [sample.s_m for sample in baseline.samples[:-1]],
            dtype=np.float64,
        )
        recomputed = spline(np.mod(reference_s, baseline.length_m))
        for index in range(baseline.unique_sample_count):
            if abs(path.offset_samples_m[index] - float(recomputed[index])) > _OFFSET_TOL:
                _add_issue(
                    issues,
                    "SPLINE_OFFSET_MISMATCH",
                    "sampled offset does not match periodic spline controls",
                    sample_index=index,
                )
    except Exception as exc:
        _add_issue(issues, "SPLINE_EVALUATION_FAILED", str(exc))
        return

    try:
        bounds = build_offset_bounds(track, baseline, path.boundary_margin_m)
        for index in range(baseline.unique_sample_count):
            offset = path.offset_samples_m[index]
            if offset < bounds.lower_m[index] - _OFFSET_TOL:
                _add_issue(
                    issues,
                    "OFFSET_BOUNDS_EXCEEDED",
                    "racing-line offset is below the feasible corridor bound",
                    sample_index=index,
                )
            if offset > bounds.upper_m[index] + _OFFSET_TOL:
                _add_issue(
                    issues,
                    "OFFSET_BOUNDS_EXCEEDED",
                    "racing-line offset is above the feasible corridor bound",
                    sample_index=index,
                )
    except BoundaryOffsetError as exc:
        _add_issue(issues, "OFFSET_BOUNDS_INVALID", str(exc))
    except Exception as exc:
        _add_issue(issues, "OFFSET_BOUNDS_CHECK_FAILED", str(exc))

    for index in range(baseline.unique_sample_count):
        sample = baseline.samples[index]
        offset = path.offset_samples_m[index]
        fold_measure = 1.0 - sample.curvature_1_per_m * offset
        if fold_measure <= _LOOP_TOL:
            _add_issue(
                issues,
                "OFFSET_FOLD",
                "lateral offset crosses the centreline curvature fold",
                sample_index=index,
            )


def _check_geometry(
    track: Track,
    baseline: BaselinePath,
    path: RacingLinePath,
    issues: list[FeasibilityIssue],
) -> None:
    unique_positions = [(sample.x_m, sample.y_m) for sample in path.samples[:-1]]
    try:
        ring = LinearRing(unique_positions)
        if not ring.is_simple:
            _add_issue(
                issues,
                "RACING_LINE_SELF_INTERSECTION",
                "racing-line path is self-intersecting",
            )
    except Exception as exc:
        _add_issue(issues, "RACING_LINE_GEOMETRY_INVALID", str(exc))

    if len(path.samples) != baseline.unique_sample_count + 1:
        _add_issue(
            issues,
            "RACING_LINE_GEOMETRY_COUNT_INVALID",
            "racing-line geometry does not have the expected sample count",
        )
        return
    centerline = np.asarray(
        [(sample.x_m, sample.y_m) for sample in baseline.samples[:-1]],
        dtype=np.float64,
    )
    normals = np.asarray(
        [(-sample.tangent_y, sample.tangent_x) for sample in baseline.samples[:-1]],
        dtype=np.float64,
    )
    offsets = np.asarray(path.offset_samples_m[:-1], dtype=np.float64)
    expected = centerline + offsets[:, None] * normals
    for index in range(baseline.unique_sample_count):
        sample = path.samples[index]
        if (
            abs(sample.x_m - float(expected[index, 0])) > _OFFSET_TOL
            or abs(sample.y_m - float(expected[index, 1])) > _OFFSET_TOL
        ):
            _add_issue(
                issues,
                "RACING_LINE_GEOMETRY_MISMATCH",
                "racing-line sample does not match centreline plus offset",
                sample_index=index,
            )

    try:
        normalized, _ = normalize_track(track)
        outer = as_array(normalized.boundaries.outer.points)
        inner = as_array(normalized.boundaries.inner.points)
        outer_line = LineString(np.vstack((outer, outer[0])))
        inner_line = LineString(np.vstack((inner, inner[0])))
        path_line = LineString([(sample.x_m, sample.y_m) for sample in path.samples])
        clearance = min(path_line.distance(outer_line), path_line.distance(inner_line))
        if clearance < path.boundary_margin_m - _MARGIN_TOL:
            _add_issue(
                issues,
                "BOUNDARY_CLEARANCE_VIOLATION",
                f"racing-line clearance {clearance:.6f} m is below margin "
                f"{path.boundary_margin_m:.6f} m",
            )
    except Exception as exc:
        _add_issue(issues, "BOUNDARY_CLEARANCE_CHECK_FAILED", str(exc))


def validate_racing_line(
    track: Track,
    baseline: BaselinePath,
    path: RacingLinePath,
    profile: SpeedProfile,
    limits: PlanningLimits,
    *,
    expected_boundary_margin_m: float | None = None,
) -> RacingLineValidationReport:
    """Validate a racing line independently from the optimiser's constraints.

    When ``expected_boundary_margin_m`` is supplied, the path must carry that
    exact configured margin; a candidate may not weaken its own clearance.
    """

    m4_report = validate_speed_profile(track, path.as_baseline_path(), profile, limits)
    issues = list(m4_report.issues)
    if expected_boundary_margin_m is not None:
        if not (math.isfinite(expected_boundary_margin_m) and expected_boundary_margin_m > 0.0):
            _add_issue(
                issues,
                "EXPECTED_BOUNDARY_MARGIN_INVALID",
                "expected boundary margin must be positive and finite",
            )
        elif abs(path.boundary_margin_m - expected_boundary_margin_m) > _MARGIN_TOL:
            _add_issue(
                issues,
                "BOUNDARY_MARGIN_MISMATCH",
                "racing-line boundary margin differs from the configured margin",
            )
    _check_offsets(track, baseline, path, issues)
    _check_geometry(track, baseline, path, issues)
    feasible = m4_report.feasible and not any(issue.severity == "error" for issue in issues)
    lap_time = m4_report.recomputed_lap_time_s
    return RacingLineValidationReport(
        feasible=feasible,
        issues=tuple(issues),
        m4_report=m4_report,
        lap_time_s=lap_time,
    )
