from __future__ import annotations

import math
from dataclasses import replace

from optiline.core.planning import (
    PathSample,
    RacingLinePath,
    baseline_fingerprint,
    build_racing_line_path,
    build_speed_profile,
    validate_racing_line,
)


def _validate_candidate(
    track,
    baseline,
    path,
    limits,
):
    profile = build_speed_profile(path.as_baseline_path(), limits)
    return validate_racing_line(track, baseline, path, profile, limits)


def _self_intersecting_path(circle_track, circle_path, planning_limits) -> RacingLinePath:
    points = [(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0), (0.0, 0.0)]
    intervals = [
        math.hypot(
            points[index + 1][0] - points[index][0],
            points[index + 1][1] - points[index][1],
        )
        for index in range(len(points) - 1)
    ]
    s_values = [0.0]
    for interval in intervals:
        s_values.append(s_values[-1] + interval)
    tangents = [
        (
            (points[index + 1][0] - points[index][0]) / intervals[index],
            (points[index + 1][1] - points[index][1]) / intervals[index],
        )
        for index in range(len(intervals))
    ]
    tangents.append(tangents[0])
    samples = tuple(
        PathSample(
            x_m=points[index][0],
            y_m=points[index][1],
            s_m=s_values[index],
            tangent_x=tangents[index][0],
            tangent_y=tangents[index][1],
            curvature_1_per_m=0.0,
        )
        for index in range(len(points))
    )
    path = RacingLinePath(
        schema_version=1,
        source_track_id=circle_track.track_id,
        source_track_name=circle_track.name,
        reference_baseline_id=baseline_fingerprint(circle_path),
        length_m=s_values[-1],
        unique_sample_count=4,
        closed=True,
        samples=samples,
        offset_samples_m=(0.0, 0.0, 0.0, 0.0, 0.0),
        control_offsets_m=(0.0, 0.0, 0.0, 0.0),
        boundary_margin_m=0.5,
    )
    return path


def test_zero_offset_candidate_is_valid(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    path = build_racing_line_path(
        circle_track,
        circle_path,
        tuple(0.0 for _ in range(16)),
        0.5,
    )
    report = _validate_candidate(circle_track, circle_path, path, planning_limits)
    assert report.feasible
    assert report.issues == ()


def test_boundary_clearance_violation_is_rejected(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    path = build_racing_line_path(
        circle_track,
        circle_path,
        tuple(9.6 for _ in range(16)),
        0.5,
    )
    report = _validate_candidate(circle_track, circle_path, path, planning_limits)
    assert not report.feasible
    codes = {issue.code for issue in report.issues}
    assert "BOUNDARY_CLEARANCE_VIOLATION" in codes
    assert "OFFSET_BOUNDS_EXCEEDED" in codes


def test_spline_overshoot_beyond_bounds_is_rejected(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    controls = tuple(9.5 if index == 8 else -9.5 for index in range(16))
    path = build_racing_line_path(circle_track, circle_path, controls, 0.5)
    report = _validate_candidate(circle_track, circle_path, path, planning_limits)
    assert not report.feasible
    assert "OFFSET_BOUNDS_EXCEEDED" in {issue.code for issue in report.issues}


def test_offset_folding_is_rejected(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    path = build_racing_line_path(
        circle_track,
        circle_path,
        tuple(60.0 for _ in range(16)),
        0.5,
    )
    report = _validate_candidate(circle_track, circle_path, path, planning_limits)
    assert not report.feasible
    assert "OFFSET_FOLD" in {issue.code for issue in report.issues}


def test_spline_control_provenance_mismatch_is_rejected(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    path = build_racing_line_path(
        circle_track,
        circle_path,
        tuple(0.0 for _ in range(16)),
        0.5,
    )
    corrupted = replace(path, control_offsets_m=tuple(1.0 for _ in range(16)))
    report = _validate_candidate(circle_track, circle_path, corrupted, planning_limits)
    assert not report.feasible
    assert "SPLINE_OFFSET_MISMATCH" in {issue.code for issue in report.issues}


def test_baseline_identity_mismatch_is_rejected(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    path = build_racing_line_path(
        circle_track,
        circle_path,
        tuple(0.0 for _ in range(16)),
        0.5,
    )
    corrupted = replace(path, reference_baseline_id="wrong-baseline")
    report = _validate_candidate(circle_track, circle_path, corrupted, planning_limits)
    assert not report.feasible
    assert "RACING_LINE_BASELINE_MISMATCH" in {issue.code for issue in report.issues}


def test_stored_margin_cannot_weaken_configured_clearance(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    path = build_racing_line_path(
        circle_track,
        circle_path,
        tuple(0.0 for _ in range(16)),
        0.5,
    )
    weaker = replace(path, boundary_margin_m=0.1)
    profile = build_speed_profile(weaker.as_baseline_path(), planning_limits)
    report = validate_racing_line(
        circle_track,
        circle_path,
        weaker,
        profile,
        planning_limits,
        expected_boundary_margin_m=0.5,
    )
    assert not report.feasible
    assert "BOUNDARY_MARGIN_MISMATCH" in {issue.code for issue in report.issues}


def test_self_intersecting_candidate_is_rejected(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    path = _self_intersecting_path(circle_track, circle_path, planning_limits)
    report = _validate_candidate(circle_track, circle_path, path, planning_limits)
    assert not report.feasible
    assert "RACING_LINE_SELF_INTERSECTION" in {issue.code for issue in report.issues}


def test_corrupted_candidate_geometry_is_rejected(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    path = build_racing_line_path(
        circle_track,
        circle_path,
        tuple(0.0 for _ in range(16)),
        0.5,
    )
    samples = list(path.samples)
    samples[2] = replace(samples[2], x_m=samples[2].x_m + 1.0)
    corrupted = replace(path, samples=tuple(samples))
    report = _validate_candidate(circle_track, circle_path, corrupted, planning_limits)
    assert not report.feasible
    assert "RACING_LINE_GEOMETRY_MISMATCH" in {issue.code for issue in report.issues}
