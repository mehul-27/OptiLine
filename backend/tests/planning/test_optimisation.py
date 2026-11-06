from __future__ import annotations

import pytest

from optiline.core.planning import (
    RacingLineOptimisationConfig,
    optimise_racing_line,
)


def test_circle_min_curvature_candidate_is_slower_and_baseline_wins(
    circle_track,
    circle_path,
    planning_limits,
    optimisation_config,
) -> None:
    result = optimise_racing_line(
        circle_track,
        circle_path,
        planning_limits,
        optimisation_config,
    )
    assert result.selected == "baseline"
    assert not result.improvement
    assert result.candidate_path is not None
    assert result.candidate_lap_time_s is not None
    assert result.candidate_lap_time_s > result.baseline_lap_time_s
    assert result.baseline_path is circle_path
    assert result.failure_reason == "no validated lap-time improvement"


def test_oval_optimisation_selects_only_independently_validated_improvement(
    oval_track,
    oval_path,
    planning_limits,
    optimisation_config,
) -> None:
    result = optimise_racing_line(
        oval_track,
        oval_path,
        planning_limits,
        optimisation_config,
    )
    assert result.selected == "racing_line"
    assert result.improvement
    assert result.candidate_path is not None
    assert result.candidate_profile is not None
    assert result.candidate_report is not None
    assert result.candidate_report.feasible
    assert result.candidate_lap_time_s is not None
    assert result.candidate_lap_time_s < result.baseline_lap_time_s
    assert result.candidate_lap_time_s == pytest.approx(result.candidate_report.lap_time_s)
    assert result.baseline_path is oval_path


def test_optimisation_is_deterministic(
    oval_track,
    oval_path,
    planning_limits,
    optimisation_config,
) -> None:
    first = optimise_racing_line(oval_track, oval_path, planning_limits, optimisation_config)
    second = optimise_racing_line(oval_track, oval_path, planning_limits, optimisation_config)
    assert first.selected == second.selected
    assert first.improvement == second.improvement
    assert first.candidate_lap_time_s is not None
    assert second.candidate_lap_time_s is not None
    assert first.candidate_lap_time_s == pytest.approx(second.candidate_lap_time_s)
    assert first.candidate_path is not None
    assert second.candidate_path is not None
    assert first.candidate_path.control_offsets_m == second.candidate_path.control_offsets_m
    assert first.candidate_path.offset_samples_m == second.candidate_path.offset_samples_m


def test_solver_failure_falls_back_to_baseline(
    oval_track,
    oval_path,
    planning_limits,
) -> None:
    config = RacingLineOptimisationConfig(
        control_point_count=16,
        boundary_margin_m=0.5,
        max_iterations=1,
        ftol=1e-7,
    )
    result = optimise_racing_line(oval_track, oval_path, planning_limits, config)
    assert result.selected == "baseline"
    assert not result.improvement
    assert result.failure_reason is not None
    assert result.optimiser_metadata is not None
    assert result.optimiser_metadata.max_iterations == 1


def test_infeasible_boundary_margin_falls_back_without_candidate(
    narrow_track,
    narrow_path,
    planning_limits,
) -> None:
    config = RacingLineOptimisationConfig(
        control_point_count=8,
        boundary_margin_m=1.5,
        max_iterations=20,
        ftol=1e-7,
    )
    result = optimise_racing_line(narrow_track, narrow_path, planning_limits, config)
    assert result.selected == "baseline"
    assert not result.improvement
    assert result.candidate_path is None
    assert result.candidate_lap_time_s is None
    assert result.failure_reason is not None
    assert "offset" in result.failure_reason.lower()


def test_asymmetric_circuit_never_accepts_invalid_or_slower_candidate(
    asymmetric_track,
    asymmetric_path,
    planning_limits,
) -> None:
    config = RacingLineOptimisationConfig(
        control_point_count=16,
        boundary_margin_m=0.5,
        max_iterations=200,
        ftol=1e-7,
    )
    result = optimise_racing_line(asymmetric_track, asymmetric_path, planning_limits, config)
    assert result.selected in ("baseline", "racing_line")
    if result.improvement:
        assert result.candidate_report is not None
        assert result.candidate_report.feasible
        assert result.candidate_lap_time_s is not None
        assert result.candidate_lap_time_s < result.baseline_lap_time_s
    else:
        assert result.selected == "baseline"


def test_optimiser_metadata_is_recorded(
    circle_track,
    circle_path,
    planning_limits,
    optimisation_config,
) -> None:
    result = optimise_racing_line(
        circle_track,
        circle_path,
        planning_limits,
        optimisation_config,
    )
    metadata = result.optimiser_metadata
    assert metadata is not None
    assert metadata.method == "SLSQP"
    assert metadata.scipy_version
    assert metadata.control_point_count == optimisation_config.control_point_count
    assert metadata.boundary_margin_m == optimisation_config.boundary_margin_m
    assert metadata.max_iterations == optimisation_config.max_iterations
    assert metadata.ftol == optimisation_config.ftol
    assert metadata.lap_time_improvement_threshold_s == pytest.approx(
        optimisation_config.lap_time_improvement_threshold_s,
    )
    assert metadata.start_strategy == "centerline_zero"


def test_improvement_threshold_blocks_small_feasible_gain(
    oval_track,
    oval_path,
    planning_limits,
    optimisation_config,
) -> None:
    config = optimisation_config.model_copy(
        update={"lap_time_improvement_threshold_s": 1.0},
    )
    result = optimise_racing_line(oval_track, oval_path, planning_limits, config)
    assert result.candidate_report is not None
    assert result.candidate_report.feasible
    assert result.candidate_lap_time_s is not None
    assert result.candidate_lap_time_s < result.baseline_lap_time_s
    assert result.baseline_lap_time_s - result.candidate_lap_time_s < 1.0
    assert result.selected == "baseline"
    assert not result.improvement
    assert result.failure_reason == "no validated lap-time improvement"
