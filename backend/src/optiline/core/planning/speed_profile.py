"""Closed-loop forward/backward speed-profile construction for M4.1."""

from __future__ import annotations

import math

import numpy as np

from optiline.core.planning.config import PLANNING_SCHEMA_VERSION, PlanningLimits
from optiline.core.planning.contracts import BaselinePath, SpeedProfile, SpeedProfileSample
from optiline.core.planning.lap_time import compute_lap_time

_CONVERGENCE_TOL = 1e-12


def build_speed_profile(
    path: BaselinePath,
    limits: PlanningLimits,
    *,
    max_iterations: int = 1000,
) -> SpeedProfile:
    """Build a feasible flying-lap speed profile by cyclic passes.

    Curvature caps and maximum speed initialize each unique sample. Forward
    acceleration and backward braking passes then run around the closed loop
    until no speed changes. The final sample is the repeated endpoint, and the
    final interval is included in acceleration, braking and lap-time
    calculations.
    """

    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise ValueError("max_iterations must be a positive integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not path.closed:
        raise ValueError("speed-profile construction requires a closed path")
    unique_count = path.unique_sample_count
    if len(path.samples) != unique_count + 1:
        raise ValueError("path must contain unique_sample_count + 1 samples")

    interval_lengths = np.asarray(
        [path.samples[index + 1].s_m - path.samples[index].s_m for index in range(unique_count)],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(interval_lengths)) or np.any(interval_lengths <= 0.0):
        raise ValueError("path intervals must be positive and finite")

    curvature = np.asarray(
        [abs(path.samples[index].curvature_1_per_m) for index in range(unique_count)],
        dtype=np.float64,
    )
    speeds = np.full(unique_count, limits.max_speed_mps, dtype=np.float64)
    curved = curvature > 0.0
    speeds[curved] = np.sqrt(limits.max_lateral_acceleration_mps2 / curvature[curved])
    speeds = np.minimum(speeds, limits.max_speed_mps)
    if np.any(speeds <= 0.0) or np.any(~np.isfinite(speeds)):
        raise ValueError("initial speed caps must be positive and finite")

    acceleration_limit = limits.max_longitudinal_acceleration_mps2
    braking_limit = limits.max_braking_deceleration_mps2
    for _ in range(max_iterations):
        changed = False

        # Acceleration pass around the closed loop.
        for index in range(unique_count):
            next_index = (index + 1) % unique_count
            candidate = math.sqrt(
                float(
                    speeds[index] * speeds[index]
                    + 2.0 * acceleration_limit * interval_lengths[index]
                ),
            )
            if candidate < float(speeds[next_index]) - _CONVERGENCE_TOL:
                speeds[next_index] = candidate
                changed = True

        # Braking pass around the closed loop.
        for index in range(unique_count - 1, -1, -1):
            next_index = (index + 1) % unique_count
            candidate = math.sqrt(
                float(
                    speeds[next_index] * speeds[next_index]
                    + 2.0 * braking_limit * interval_lengths[index]
                ),
            )
            if candidate < float(speeds[index]) - _CONVERGENCE_TOL:
                speeds[index] = candidate
                changed = True

        if not changed:
            break
    else:
        raise ValueError("speed profile did not converge within max_iterations")

    closed_speeds = np.concatenate((speeds, speeds[:1]))
    interval_accelerations = (
        closed_speeds[1:] * closed_speeds[1:] - closed_speeds[:-1] * closed_speeds[:-1]
    ) / (2.0 * interval_lengths)
    if np.any(~np.isfinite(interval_accelerations)):
        raise ValueError("speed profile produced a non-finite interval acceleration")

    timing = compute_lap_time(path, closed_speeds.tolist())
    samples = tuple(
        SpeedProfileSample(
            s_m=path.samples[index].s_m,
            speed_mps=float(closed_speeds[index]),
            curvature_1_per_m=path.samples[index].curvature_1_per_m,
            lateral_acceleration_mps2=float(
                closed_speeds[index]
                * closed_speeds[index]
                * abs(path.samples[index].curvature_1_per_m),
            ),
        )
        for index in range(len(closed_speeds))
    )
    return SpeedProfile(
        schema_version=PLANNING_SCHEMA_VERSION,
        source_track_id=path.source_track_id,
        path_length_m=path.length_m,
        unique_sample_count=unique_count,
        closed=True,
        samples=samples,
        interval_longitudinal_accelerations_mps2=tuple(
            float(value) for value in interval_accelerations
        ),
        lap_time_s=timing.lap_time_s,
        limits=limits,
    )
