"""Deterministic minimum-curvature racing-line optimisation for M4.2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from scipy.optimize import minimize

from optiline.core.planning.config import PLANNING_SCHEMA_VERSION, PlanningLimits
from optiline.core.planning.contracts import BaselinePath, SpeedProfile
from optiline.core.planning.feasibility import FeasibilityReport, validate_speed_profile
from optiline.core.planning.racing_line import (
    BoundaryOffsetError,
    RacingLineOptimiserMetadata,
    RacingLinePath,
    build_offset_bounds,
    build_racing_line_path,
    evaluate_periodic_offsets,
)
from optiline.core.planning.racing_line_validation import (
    RacingLineValidationReport,
    validate_racing_line,
)
from optiline.core.planning.speed_profile import build_speed_profile
from optiline.core.track.schema import Track

_DEFAULT_LAP_TIME_IMPROVEMENT_THRESHOLD_S = 0.01
_INVALID_OBJECTIVE_PENALTY = 1e12


class RacingLineOptimisationConfig(BaseModel):
    """Configuration for deterministic constrained racing-line optimisation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictInt = PLANNING_SCHEMA_VERSION
    control_point_count: int = Field(default=24, ge=4, le=128)
    boundary_margin_m: float = Field(default=0.5, gt=0.0, allow_inf_nan=False)
    max_iterations: int = Field(default=200, gt=0)
    ftol: float = Field(default=1e-9, gt=0.0, allow_inf_nan=False)
    lap_time_improvement_threshold_s: float = Field(
        default=_DEFAULT_LAP_TIME_IMPROVEMENT_THRESHOLD_S,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "Reporting tolerance in seconds. A candidate is a meaningful "
            "improvement only when candidate_lap_time < baseline_lap_time - "
            "lap_time_improvement_threshold_s. This is not a physical parameter."
        ),
    )

    @model_validator(mode="after")
    def validate_config(self) -> RacingLineOptimisationConfig:
        if self.schema_version != PLANNING_SCHEMA_VERSION:
            raise ValueError(f"unsupported racing-line schema_version: {self.schema_version}")
        return self


@dataclass(frozen=True)
class RacingLineResult:
    """Baseline/candidate comparison with independently validated lap times."""

    baseline_path: BaselinePath
    baseline_profile: SpeedProfile
    baseline_report: FeasibilityReport
    baseline_lap_time_s: float
    candidate_path: RacingLinePath | None
    candidate_profile: SpeedProfile | None
    candidate_report: RacingLineValidationReport | None
    candidate_lap_time_s: float | None
    selected: Literal["baseline", "racing_line"]
    improvement: bool
    failure_reason: str | None
    optimiser_metadata: RacingLineOptimiserMetadata | None
    config: RacingLineOptimisationConfig


def _baseline_result(
    *,
    baseline_path: BaselinePath,
    baseline_profile: SpeedProfile,
    baseline_report: FeasibilityReport,
    baseline_lap_time_s: float,
    failure_reason: str | None,
    optimiser_metadata: RacingLineOptimiserMetadata | None,
    config: RacingLineOptimisationConfig,
    candidate_path: RacingLinePath | None = None,
    candidate_profile: SpeedProfile | None = None,
    candidate_report: RacingLineValidationReport | None = None,
    candidate_lap_time_s: float | None = None,
) -> RacingLineResult:
    return RacingLineResult(
        baseline_path=baseline_path,
        baseline_profile=baseline_profile,
        baseline_report=baseline_report,
        baseline_lap_time_s=baseline_lap_time_s,
        candidate_path=candidate_path,
        candidate_profile=candidate_profile,
        candidate_report=candidate_report,
        candidate_lap_time_s=candidate_lap_time_s,
        selected="baseline",
        improvement=False,
        failure_reason=failure_reason,
        optimiser_metadata=optimiser_metadata,
        config=config,
    )


def optimise_racing_line(
    track: Track,
    baseline_path: BaselinePath,
    limits: PlanningLimits,
    config: RacingLineOptimisationConfig,
) -> RacingLineResult:
    """Run deterministic minimum-curvature SLSQP and validate every candidate.

    The optimiser objective is geometric curvature energy, not lap time. The
    M4.1 speed profile and independent validator decide whether a candidate is
    feasible and whether it is faster than the baseline.
    """

    baseline_profile = build_speed_profile(baseline_path, limits)
    baseline_report = validate_speed_profile(
        track,
        baseline_path,
        baseline_profile,
        limits,
    )
    if not baseline_report.feasible or baseline_report.recomputed_lap_time_s is None:
        raise ValueError("M4.1 baseline path/profile is not independently feasible")
    baseline_lap_time = baseline_report.recomputed_lap_time_s

    try:
        bounds = build_offset_bounds(track, baseline_path, config.boundary_margin_m)
    except BoundaryOffsetError as exc:
        return _baseline_result(
            baseline_path=baseline_path,
            baseline_profile=baseline_profile,
            baseline_report=baseline_report,
            baseline_lap_time_s=baseline_lap_time,
            failure_reason=f"offset bounds unavailable: {exc}",
            optimiser_metadata=None,
            config=config,
        )
    except Exception as exc:
        return _baseline_result(
            baseline_path=baseline_path,
            baseline_profile=baseline_profile,
            baseline_report=baseline_report,
            baseline_lap_time_s=baseline_lap_time,
            failure_reason=f"offset bounds failed: {exc}",
            optimiser_metadata=None,
            config=config,
        )

    unique_count = baseline_path.unique_sample_count
    reference_s = np.asarray(
        [baseline_path.samples[index].s_m for index in range(unique_count)],
        dtype=np.float64,
    )
    lower = np.asarray(bounds.lower_m[:unique_count], dtype=np.float64)
    upper = np.asarray(bounds.upper_m[:unique_count], dtype=np.float64)
    initial = np.zeros(config.control_point_count, dtype=np.float64)
    initial_constraint = np.concatenate((-lower, upper))
    if np.any(initial_constraint < -1e-9):
        return _baseline_result(
            baseline_path=baseline_path,
            baseline_profile=baseline_profile,
            baseline_report=baseline_report,
            baseline_lap_time_s=baseline_lap_time,
            failure_reason="centerline offset is outside the feasible boundary margin",
            optimiser_metadata=None,
            config=config,
        )

    def offset_constraint(values: np.ndarray) -> np.ndarray:
        try:
            offsets = evaluate_periodic_offsets(
                tuple(float(value) for value in values),
                baseline_path.length_m,
                reference_s,
            )
            return np.concatenate((offsets - lower, upper - offsets))
        except Exception:
            return np.full(2 * unique_count, -_INVALID_OBJECTIVE_PENALTY)

    def objective(values: np.ndarray) -> float:
        try:
            path = build_racing_line_path(
                track,
                baseline_path,
                tuple(float(value) for value in values),
                config.boundary_margin_m,
            )
            curvature = np.asarray(
                [sample.curvature_1_per_m for sample in path.samples[:-1]],
                dtype=np.float64,
            )
            intervals = np.asarray(
                [
                    path.samples[index + 1].s_m - path.samples[index].s_m
                    for index in range(unique_count)
                ],
                dtype=np.float64,
            )
            value = float(np.sum(curvature * curvature * intervals))
            return value if math.isfinite(value) else _INVALID_OBJECTIVE_PENALTY
        except Exception:
            return _INVALID_OBJECTIVE_PENALTY

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        constraints=({"type": "ineq", "fun": offset_constraint},),
        options={
            "maxiter": config.max_iterations,
            "ftol": config.ftol,
            "disp": False,
        },
    )
    metadata = RacingLineOptimiserMetadata(
        method="SLSQP",
        scipy_version=scipy.__version__,
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
        objective_value=float(result.fun),
        control_point_count=config.control_point_count,
        boundary_margin_m=config.boundary_margin_m,
        max_iterations=config.max_iterations,
        ftol=config.ftol,
        lap_time_improvement_threshold_s=config.lap_time_improvement_threshold_s,
        start_strategy="centerline_zero",
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return _baseline_result(
            baseline_path=baseline_path,
            baseline_profile=baseline_profile,
            baseline_report=baseline_report,
            baseline_lap_time_s=baseline_lap_time,
            failure_reason=f"SLSQP did not converge: {result.message}",
            optimiser_metadata=metadata,
            config=config,
        )

    try:
        candidate = build_racing_line_path(
            track,
            baseline_path,
            tuple(float(value) for value in result.x),
            config.boundary_margin_m,
            optimiser_metadata=metadata,
        )
    except Exception as exc:
        return _baseline_result(
            baseline_path=baseline_path,
            baseline_profile=baseline_profile,
            baseline_report=baseline_report,
            baseline_lap_time_s=baseline_lap_time,
            failure_reason=f"candidate path construction failed: {exc}",
            optimiser_metadata=metadata,
            config=config,
        )

    try:
        candidate_profile = build_speed_profile(candidate.as_baseline_path(), limits)
        candidate_report = validate_racing_line(
            track,
            baseline_path,
            candidate,
            candidate_profile,
            limits,
            expected_boundary_margin_m=config.boundary_margin_m,
        )
    except Exception as exc:
        return _baseline_result(
            baseline_path=baseline_path,
            baseline_profile=baseline_profile,
            baseline_report=baseline_report,
            baseline_lap_time_s=baseline_lap_time,
            failure_reason=f"candidate validation failed: {exc}",
            optimiser_metadata=metadata,
            config=config,
            candidate_path=candidate,
        )

    candidate_lap_time = candidate_report.lap_time_s
    if not candidate_report.feasible or candidate_lap_time is None:
        codes = sorted({issue.code for issue in candidate_report.issues})
        reason = ", ".join(codes) if codes else "candidate is not feasible"
        return _baseline_result(
            baseline_path=baseline_path,
            baseline_profile=baseline_profile,
            baseline_report=baseline_report,
            baseline_lap_time_s=baseline_lap_time,
            failure_reason=f"candidate rejected: {reason}",
            optimiser_metadata=metadata,
            config=config,
            candidate_path=candidate,
            candidate_profile=candidate_profile,
            candidate_report=candidate_report,
            candidate_lap_time_s=candidate_lap_time,
        )

    improved = candidate_lap_time < (baseline_lap_time - config.lap_time_improvement_threshold_s)
    return RacingLineResult(
        baseline_path=baseline_path,
        baseline_profile=baseline_profile,
        baseline_report=baseline_report,
        baseline_lap_time_s=baseline_lap_time,
        candidate_path=candidate,
        candidate_profile=candidate_profile,
        candidate_report=candidate_report,
        candidate_lap_time_s=candidate_lap_time,
        selected="racing_line" if improved else "baseline",
        improvement=improved,
        failure_reason=None if improved else "no validated lap-time improvement",
        optimiser_metadata=metadata,
        config=config,
    )
