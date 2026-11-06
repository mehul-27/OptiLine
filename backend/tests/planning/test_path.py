from __future__ import annotations

import math

import pytest

from optiline.core.planning import (
    build_baseline_path,
    build_speed_profile,
    validate_speed_profile,
)
from tests.fixtures.track_builders import (
    make_circle_track,
    make_self_intersecting_track,
    make_uncalibrated_track,
)


def test_circle_path_is_closed_and_has_final_interval(circle_path) -> None:
    assert circle_path.source_track_id == "circle-r50"
    assert circle_path.unique_sample_count == 360
    assert len(circle_path.samples) == 361
    assert circle_path.length_m == pytest.approx(math.tau * 50.0, rel=5e-5)
    first = circle_path.samples[0]
    last = circle_path.samples[-1]
    assert first.x_m == pytest.approx(last.x_m)
    assert first.y_m == pytest.approx(last.y_m)
    assert first.s_m == pytest.approx(0.0)
    assert last.s_m == pytest.approx(circle_path.length_m)
    assert last.s_m - circle_path.samples[-2].s_m > 0.0
    assert first.x_m == pytest.approx(50.0, abs=1e-9)
    assert first.y_m == pytest.approx(0.0, abs=1e-9)
    assert math.hypot(first.tangent_x, first.tangent_y) == pytest.approx(1.0)


def test_circle_path_curvature_matches_analytic_radius(circle_path) -> None:
    curvature = [abs(sample.curvature_1_per_m) for sample in circle_path.samples[:-1]]
    assert sum(curvature) / len(curvature) == pytest.approx(1.0 / 50.0, rel=1e-3)
    assert all(value > 0.0 for value in curvature)


def test_oval_path_length_matches_analytic_stadium(oval_path) -> None:
    expected = 2.0 * 100.0 + math.tau * 30.0
    assert oval_path.length_m == pytest.approx(expected, rel=1e-4)
    assert oval_path.unique_sample_count == 720
    assert len(oval_path.samples) == 721
    assert oval_path.samples[0].x_m == pytest.approx(
        oval_path.samples[-1].x_m,
    )
    assert oval_path.samples[0].y_m == pytest.approx(
        oval_path.samples[-1].y_m,
    )


def test_path_construction_is_deterministic() -> None:
    track = make_circle_track(count=360)
    first = build_baseline_path(track, sample_count=360)
    second = build_baseline_path(track, sample_count=360)
    assert first == second


def test_invalid_or_uncalibrated_track_is_rejected() -> None:
    with pytest.raises(ValueError, match="not simulation-ready"):
        build_baseline_path(make_self_intersecting_track(), sample_count=128)
    with pytest.raises(ValueError, match="not simulation-ready"):
        build_baseline_path(make_uncalibrated_track(), sample_count=128)


def test_invalid_sample_count_is_rejected() -> None:
    track = make_circle_track(count=64)
    with pytest.raises(ValueError, match="sample_count"):
        build_baseline_path(track, sample_count=3)
    with pytest.raises(ValueError, match="sample_count"):
        build_baseline_path(track, sample_count=True)  # type: ignore[arg-type]


def test_minimum_supported_sample_count_produces_closed_arrays(
    circle_track,
    planning_limits,
) -> None:
    path = build_baseline_path(circle_track, sample_count=4)
    assert path.unique_sample_count == 4
    assert len(path.samples) == 5
    assert path.samples[0].x_m == pytest.approx(path.samples[-1].x_m)
    assert path.samples[0].y_m == pytest.approx(path.samples[-1].y_m)
    assert path.samples[-1].s_m - path.samples[-2].s_m > 0.0
    profile = build_speed_profile(path, planning_limits)
    assert len(profile.samples) == 5
    assert len(profile.interval_longitudinal_accelerations_mps2) == 4
    assert profile.samples[0].speed_mps == pytest.approx(profile.samples[-1].speed_mps)

    report = validate_speed_profile(circle_track, path, profile, planning_limits)
    assert not report.feasible
    assert "PATH_OUTSIDE_CORRIDOR" in {issue.code for issue in report.issues}
