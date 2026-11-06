from __future__ import annotations

import math

import pytest

from optiline.core.planning import compute_lap_time


def test_constant_speed_lap_time_includes_final_interval(circle_path) -> None:
    speed = math.sqrt(4.0 * 50.0)
    speeds = tuple(speed for _ in circle_path.samples)
    timing = compute_lap_time(circle_path, speeds)
    assert timing.lap_time_s == pytest.approx(circle_path.length_m / speed)
    assert len(timing.interval_times_s) == circle_path.unique_sample_count
    assert len(timing.interval_lengths_m) == circle_path.unique_sample_count
    assert timing.interval_lengths_m[-1] > 0.0
    assert math.fsum(timing.interval_times_s) == pytest.approx(timing.lap_time_s)


def test_lap_time_rejects_bad_speed_arrays(circle_path) -> None:
    speeds = [10.0] * len(circle_path.samples)
    with pytest.raises(ValueError, match="match path samples"):
        compute_lap_time(circle_path, speeds[:-1])
    speeds[3] = 0.0
    with pytest.raises(ValueError, match="positive"):
        compute_lap_time(circle_path, speeds)
    speeds = [10.0] * len(circle_path.samples)
    speeds[2] = math.nan
    with pytest.raises(ValueError, match="finite"):
        compute_lap_time(circle_path, speeds)
