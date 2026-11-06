from __future__ import annotations

from dataclasses import replace

import pytest

from optiline.core.planning import (
    BaselinePath,
    PathSample,
    SpeedProfile,
    SpeedProfileSample,
    build_speed_profile,
    compute_lap_time,
    validate_speed_profile,
)
from optiline.core.track.schema import Boundary, StartFinish, TrackBoundaries
from tests.fixtures.track_builders import make_circle_track, make_uncalibrated_track


def issue_codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_validator_accepts_circle_and_oval(
    circle_track,
    circle_path,
    oval_track,
    oval_path,
    planning_limits,
) -> None:
    for track, path in ((circle_track, circle_path), (oval_track, oval_path)):
        profile = build_speed_profile(path, planning_limits)
        report = validate_speed_profile(track, path, profile, planning_limits)
        assert report.feasible
        assert report.issues == ()
        assert report.recomputed_lap_time_s == pytest.approx(profile.lap_time_s)
        assert report.max_lateral_acceleration_mps2 is not None
        assert report.max_lateral_acceleration_mps2 <= (
            planning_limits.max_lateral_acceleration_mps2 + 1e-9
        )
        assert report.max_longitudinal_acceleration_mps2 is not None
        assert report.max_longitudinal_acceleration_mps2 <= (
            planning_limits.max_longitudinal_acceleration_mps2 + 1e-9
        )
        assert report.max_braking_deceleration_mps2 is not None
        assert report.max_braking_deceleration_mps2 <= (
            planning_limits.max_braking_deceleration_mps2 + 1e-9
        )


def test_validator_recomputes_and_rejects_bad_lap_time(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    profile = build_speed_profile(circle_path, planning_limits)
    corrupted = replace(profile, lap_time_s=profile.lap_time_s + 1.0)
    report = validate_speed_profile(circle_track, circle_path, corrupted, planning_limits)
    assert not report.feasible
    assert "LAP_TIME_MISMATCH" in issue_codes(report)


def test_validator_recomputes_interval_lengths_from_geometry(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    profile = build_speed_profile(circle_path, planning_limits)
    shifted_samples = tuple(
        replace(sample, s_m=sample.s_m + 0.01) if index >= 10 else sample
        for index, sample in enumerate(circle_path.samples)
    )
    corrupted_path = replace(
        circle_path,
        samples=shifted_samples,
        length_m=circle_path.length_m + 0.01,
    )
    report = validate_speed_profile(circle_track, corrupted_path, profile, planning_limits)
    assert not report.feasible
    assert "PATH_ARC_LENGTH_MISMATCH" in issue_codes(report)


def test_validator_rejects_source_metadata_mismatch(
    circle_track,
    circle_path,
    planning_limits,
    vehicle_config,
) -> None:
    profile = build_speed_profile(circle_path, planning_limits)
    wrong_path = replace(circle_path, source_track_id="wrong-track")
    report = validate_speed_profile(circle_track, wrong_path, profile, planning_limits)
    assert not report.feasible
    assert "PATH_TRACK_MISMATCH" in issue_codes(report)

    wrong_profile = replace(profile, source_track_id="wrong-track")
    report = validate_speed_profile(circle_track, circle_path, wrong_profile, planning_limits)
    assert not report.feasible
    assert "PROFILE_TRACK_MISMATCH" in issue_codes(report)

    other_limits = type(planning_limits).from_dynamic_vehicle_config(
        vehicle_config,
        max_lateral_acceleration_mps2=3.5,
    )
    wrong_limits = replace(profile, limits=other_limits)
    report = validate_speed_profile(circle_track, circle_path, wrong_limits, planning_limits)
    assert not report.feasible
    assert "PROFILE_LIMITS_MISMATCH" in issue_codes(report)


def test_validator_rejects_profile_arc_length_mismatch(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    profile = build_speed_profile(circle_path, planning_limits)
    samples = list(profile.samples)
    samples[1] = replace(samples[1], s_m=samples[1].s_m + 0.1)
    corrupted = replace(profile, samples=tuple(samples))
    report = validate_speed_profile(circle_track, circle_path, corrupted, planning_limits)
    assert not report.feasible
    assert "PROFILE_ARC_LENGTH_MISMATCH" in issue_codes(report)


def test_validator_rejects_corrupted_path_curvature(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    zero_curvature_path = replace(
        circle_path,
        samples=tuple(replace(sample, curvature_1_per_m=0.0) for sample in circle_path.samples),
    )
    profile = build_speed_profile(zero_curvature_path, planning_limits)
    report = validate_speed_profile(
        circle_track,
        zero_curvature_path,
        profile,
        planning_limits,
    )
    assert not report.feasible
    assert "PATH_CURVATURE_MISMATCH" in issue_codes(report)


def test_validator_rejects_corrupted_path_tangent(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    profile = build_speed_profile(circle_path, planning_limits)
    samples = list(circle_path.samples)
    sample = samples[1]
    samples[1] = replace(
        sample,
        tangent_x=-sample.tangent_y,
        tangent_y=sample.tangent_x,
    )
    corrupted = replace(circle_path, samples=tuple(samples))
    report = validate_speed_profile(circle_track, corrupted, profile, planning_limits)
    assert not report.feasible
    assert "PATH_TANGENT_MISMATCH" in issue_codes(report)


def test_validator_rejects_profile_curvature_mismatch(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    profile = build_speed_profile(circle_path, planning_limits)
    sample = replace(
        profile.samples[1],
        curvature_1_per_m=profile.samples[1].curvature_1_per_m + 0.1,
    )
    corrupted = replace(
        profile,
        samples=(profile.samples[0], sample, *profile.samples[2:]),
    )
    report = validate_speed_profile(circle_track, circle_path, corrupted, planning_limits)
    assert not report.feasible
    assert "PROFILE_CURVATURE_MISMATCH" in issue_codes(report)


def test_validator_recomputes_lateral_from_path(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    profile = build_speed_profile(circle_path, planning_limits)
    sample = profile.samples[1]
    samples = (
        profile.samples[0],
        replace(sample, lateral_acceleration_mps2=sample.lateral_acceleration_mps2 + 1.0),
        *profile.samples[2:],
    )
    corrupted = replace(profile, samples=samples)
    report = validate_speed_profile(circle_track, circle_path, corrupted, planning_limits)
    assert not report.feasible
    assert "PROFILE_LATERAL_MISMATCH" in issue_codes(report)


def test_validator_recomputes_interval_acceleration(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    profile = build_speed_profile(circle_path, planning_limits)
    accelerations = list(profile.interval_longitudinal_accelerations_mps2)
    accelerations[0] += 1.0
    corrupted = replace(
        profile,
        interval_longitudinal_accelerations_mps2=tuple(accelerations),
    )
    report = validate_speed_profile(circle_track, circle_path, corrupted, planning_limits)
    assert not report.feasible
    assert "PROFILE_ACCELERATION_MISMATCH" in issue_codes(report)


def test_validator_rejects_speed_limit_violation(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    profile = build_speed_profile(circle_path, planning_limits)
    sample = replace(profile.samples[1], speed_mps=planning_limits.max_speed_mps + 5.0)
    corrupted = replace(
        profile,
        samples=(profile.samples[0], sample, *profile.samples[2:]),
    )
    report = validate_speed_profile(circle_track, circle_path, corrupted, planning_limits)
    assert not report.feasible
    assert "PROFILE_MAX_SPEED_EXCEEDED" in issue_codes(report)


def test_validator_rejects_path_outside_corridor(
    circle_track,
    circle_path,
    planning_limits,
) -> None:
    profile = build_speed_profile(circle_path, planning_limits)
    shifted = replace(
        circle_path,
        samples=tuple(replace(sample, x_m=sample.x_m + 1000.0) for sample in circle_path.samples),
    )
    report = validate_speed_profile(circle_track, shifted, profile, planning_limits)
    assert not report.feasible
    assert "PATH_OUTSIDE_CORRIDOR" in issue_codes(report)


def test_validator_rejects_uncalibrated_track(
    circle_path,
    planning_limits,
) -> None:
    profile = build_speed_profile(circle_path, planning_limits)
    report = validate_speed_profile(
        make_uncalibrated_track(),
        circle_path,
        profile,
        planning_limits,
    )
    assert not report.feasible
    assert "TRACK_NOT_SIMULATION_READY" in issue_codes(report)


def test_validator_checks_path_segments_not_only_endpoints(
    planning_limits,
) -> None:
    base = make_circle_track(count=128)
    track = base.model_copy(
        update={
            "track_id": "ring-chord",
            "name": "Ring chord",
            "boundaries": TrackBoundaries(
                outer=Boundary(points=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]),
                inner=Boundary(points=[(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)]),
            ),
            "start_finish": StartFinish(
                line=((5.0, 3.0), (5.0, 0.0)),
                forward=(1.0, 0.0),
            ),
        },
    )

    points = [(1.0, 5.0), (9.0, 5.0), (5.0, 9.0), (5.0, 1.0), (1.0, 5.0)]
    intervals = [
        8.0,
        ((5.0 - 9.0) ** 2 + (9.0 - 5.0) ** 2) ** 0.5,
        8.0,
        4.0,
    ]
    s_values = [0.0]
    for interval in intervals:
        s_values.append(s_values[-1] + interval)
    tangents = [
        (1.0, 0.0),
        ((5.0 - 9.0) / intervals[1], (9.0 - 5.0) / intervals[1]),
        (0.0, -1.0),
        (0.0, 1.0),
        (1.0, 0.0),
    ]
    samples = tuple(
        PathSample(
            x_m=point[0],
            y_m=point[1],
            s_m=s_values[index],
            tangent_x=tangents[index][0],
            tangent_y=tangents[index][1],
            curvature_1_per_m=0.0,
        )
        for index, point in enumerate(points)
    )
    path = BaselinePath(
        schema_version=1,
        source_track_id="ring-chord",
        source_track_name="Ring chord",
        length_m=s_values[-1],
        unique_sample_count=4,
        closed=True,
        samples=samples,
    )
    profile_samples = tuple(
        SpeedProfileSample(
            s_m=sample.s_m,
            speed_mps=5.0,
            curvature_1_per_m=0.0,
            lateral_acceleration_mps2=0.0,
        )
        for sample in samples
    )
    lap_time = compute_lap_time(path, [5.0] * len(samples)).lap_time_s
    profile = SpeedProfile(
        schema_version=1,
        source_track_id="ring-chord",
        path_length_m=path.length_m,
        unique_sample_count=4,
        closed=True,
        samples=profile_samples,
        interval_longitudinal_accelerations_mps2=(0.0, 0.0, 0.0, 0.0),
        lap_time_s=lap_time,
        limits=planning_limits,
    )

    report = validate_speed_profile(track, path, profile, planning_limits)
    assert not report.feasible
    assert "PATH_OUTSIDE_CORRIDOR" in issue_codes(report)
