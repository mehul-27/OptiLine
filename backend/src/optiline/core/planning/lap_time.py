"""Deterministic flying-lap timing from a closed path and speeds."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from optiline.core.planning.contracts import BaselinePath


@dataclass(frozen=True)
class LapTimingResult:
    """Lap time and interval contributions for a closed profile."""

    lap_time_s: float
    interval_lengths_m: tuple[float, ...]
    interval_times_s: tuple[float, ...]


def compute_lap_time(path: BaselinePath, speeds: Sequence[float]) -> LapTimingResult:
    """Compute flying-lap time using the trapezoidal interval rule.

    Speeds must include the repeated final endpoint. Every interval, including
    the final closure interval, contributes ``2 * ds / (v_i + v_{i+1})``.
    """

    if not path.closed:
        raise ValueError("lap timing requires a closed path")
    if len(speeds) != len(path.samples):
        raise ValueError("speeds must match path samples")
    if len(speeds) < 2:
        raise ValueError("at least one interval is required")

    interval_lengths: list[float] = []
    interval_times: list[float] = []
    for index in range(len(path.samples) - 1):
        ds = path.samples[index + 1].s_m - path.samples[index].s_m
        if not math.isfinite(ds) or ds <= 0.0:
            raise ValueError("path intervals must have positive finite length")
        first = float(speeds[index])
        second = float(speeds[index + 1])
        if not (math.isfinite(first) and math.isfinite(second)):
            raise ValueError("speeds must be finite")
        if first <= 0.0 or second <= 0.0:
            raise ValueError("flying-lap speeds must be positive")
        interval_lengths.append(ds)
        interval_times.append(2.0 * ds / (first + second))

    lap_time = math.fsum(interval_times)
    if not math.isfinite(lap_time) or lap_time <= 0.0:
        raise ValueError("lap time must be positive and finite")
    return LapTimingResult(
        lap_time_s=lap_time,
        interval_lengths_m=tuple(interval_lengths),
        interval_times_s=tuple(interval_times),
    )
