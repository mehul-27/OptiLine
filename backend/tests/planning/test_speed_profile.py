from __future__ import annotations

import math

import pytest

from optiline.core.planning import (
    BaselinePath,
    PathSample,
    PlanningLimits,
    build_speed_profile,
    validate_speed_profile,
)
from optiline.core.vehicle.dynamic_config import DynamicVehicleConfig


def _rectangle_path(curvature: tuple[float, float, float, float]) -> BaselinePath:
    points = [(0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0), (0.0, 0.0)]
    s_values = [0.0, 2.0, 5.0, 7.0, 10.0]
    tangents = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)]
    closed_curvature = (*curvature, curvature[0])
    samples = tuple(
        PathSample(
            x_m=point[0],
            y_m=point[1],
            s_m=s_values[index],
            tangent_x=tangents[index][0],
            tangent_y=tangents[index][1],
            curvature_1_per_m=closed_curvature[index],
        )
        for index, point in enumerate(points)
    )
    return BaselinePath(
        schema_version=1,
        source_track_id="synthetic-rectangle",
        source_track_name="Synthetic rectangle",
        length_m=s_values[-1],
        unique_sample_count=4,
        closed=True,
        samples=samples,
    )


def test_circle_profile_is_lateral_limited_and_closed(
    circle_path,
    planning_limits: PlanningLimits,
) -> None:
    profile = build_speed_profile(circle_path, planning_limits)
    expected_speed = math.sqrt(planning_limits.max_lateral_acceleration_mps2 * 50.0)
    assert len(profile.samples) == circle_path.unique_sample_count + 1
    assert len(profile.interval_longitudinal_accelerations_mps2) == circle_path.unique_sample_count
    assert profile.samples[0].speed_mps == pytest.approx(profile.samples[-1].speed_mps)
    for sample in profile.samples:
        assert sample.speed_mps == pytest.approx(expected_speed, rel=1e-9)
        assert sample.lateral_acceleration_mps2 <= (
            planning_limits.max_lateral_acceleration_mps2 + 1e-9
        )
    assert profile.lap_time_s == pytest.approx(circle_path.length_m / expected_speed)
    assert circle_path.samples[-1].s_m - circle_path.samples[-2].s_m > 0.0
    assert profile.limits == planning_limits


def test_oval_profile_respects_acceleration_and_braking(
    oval_path,
    planning_limits: PlanningLimits,
) -> None:
    profile = build_speed_profile(oval_path, planning_limits)
    speeds = [sample.speed_mps for sample in profile.samples]
    accelerations = profile.interval_longitudinal_accelerations_mps2
    assert len(accelerations) == oval_path.unique_sample_count
    assert min(speeds) < max(speeds)
    assert max(speeds) <= planning_limits.max_speed_mps + 1e-9
    assert max(accelerations) <= planning_limits.max_longitudinal_acceleration_mps2 + 1e-9
    assert min(accelerations) >= -planning_limits.max_braking_deceleration_mps2 - 1e-9
    assert max(sample.lateral_acceleration_mps2 for sample in profile.samples) <= (
        planning_limits.max_lateral_acceleration_mps2 + 1e-9
    )
    assert profile.samples[0].speed_mps == pytest.approx(profile.samples[-1].speed_mps)
    assert len(profile.interval_longitudinal_accelerations_mps2) == oval_path.unique_sample_count


def test_irregular_intervals_and_sharp_closure_peak(
    planning_limits: PlanningLimits,
) -> None:
    path = _rectangle_path((1.0, 0.0, 0.0, 0.0))
    profile = build_speed_profile(path, planning_limits)
    assert len(profile.samples) == 5
    assert len(profile.interval_longitudinal_accelerations_mps2) == 4
    assert profile.samples[0].speed_mps == pytest.approx(profile.samples[-1].speed_mps)
    assert profile.samples[0].speed_mps == pytest.approx(2.0)
    assert max(profile.interval_longitudinal_accelerations_mps2) <= (
        planning_limits.max_longitudinal_acceleration_mps2 + 1e-9
    )
    assert min(profile.interval_longitudinal_accelerations_mps2) >= (
        -planning_limits.max_braking_deceleration_mps2 - 1e-9
    )
    assert max(sample.lateral_acceleration_mps2 for sample in profile.samples) <= (
        planning_limits.max_lateral_acceleration_mps2 + 1e-9
    )


def test_profile_uses_absolute_curvature_for_sign_changes(
    planning_limits: PlanningLimits,
) -> None:
    path = _rectangle_path((1.0, -1.0, 0.0, 0.0))
    profile = build_speed_profile(path, planning_limits)
    assert profile.samples[0].speed_mps == pytest.approx(profile.samples[-1].speed_mps)
    assert all(
        sample.lateral_acceleration_mps2 <= planning_limits.max_lateral_acceleration_mps2 + 1e-9
        for sample in profile.samples
    )


def test_profile_respects_maximum_speed(
    circle_path,
    vehicle_config: DynamicVehicleConfig,
) -> None:
    slower_vehicle = vehicle_config.model_copy(update={"max_speed_mps": 10.0})
    limits = PlanningLimits.from_dynamic_vehicle_config(slower_vehicle)
    profile = build_speed_profile(circle_path, limits)
    assert all(sample.speed_mps <= 10.0 + 1e-9 for sample in profile.samples)


def test_profile_is_deterministic(
    oval_path,
    planning_limits: PlanningLimits,
) -> None:
    first = build_speed_profile(oval_path, planning_limits)
    second = build_speed_profile(oval_path, planning_limits)
    assert first == second


def test_profile_validator_accepts_generated_profiles(
    circle_track,
    circle_path,
    planning_limits: PlanningLimits,
) -> None:
    profile = build_speed_profile(circle_path, planning_limits)
    report = validate_speed_profile(circle_track, circle_path, profile, planning_limits)
    assert report.feasible
    assert report.issues == ()
    assert report.recomputed_lap_time_s == pytest.approx(profile.lap_time_s)


def test_profile_builder_rejects_invalid_iteration_limit(
    circle_path,
    planning_limits: PlanningLimits,
) -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        build_speed_profile(circle_path, planning_limits, max_iterations=0)
    with pytest.raises(ValueError, match="max_iterations"):
        build_speed_profile(circle_path, planning_limits, max_iterations=True)  # type: ignore[arg-type]


def test_profile_builder_fails_deterministically_on_non_convergence(
    oval_path,
    planning_limits: PlanningLimits,
) -> None:
    with pytest.raises(ValueError, match="did not converge"):
        build_speed_profile(oval_path, planning_limits, max_iterations=1)
