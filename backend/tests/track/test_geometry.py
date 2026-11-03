from __future__ import annotations

import math

import numpy as np
import pytest

from optiline.core.track.geometry import (
    as_array,
    closed_polyline_length,
    compute_curvature,
    derive_centerline,
    ensure_ccw,
    is_ccw,
    resample_closed_polyline,
    signed_area,
    summarize_track,
)
from optiline.core.track.schema import Track
from tests.fixtures.track_builders import make_circle_track, make_oval_track


def test_square_perimeter_and_signed_area() -> None:
    square = np.asarray([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    assert closed_polyline_length(square) == pytest.approx(4.0)
    assert signed_area(square) == pytest.approx(1.0)
    assert is_ccw(square)


def test_ensure_ccw_reverses_clockwise_input_without_mutating() -> None:
    clockwise = np.asarray([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
    original = clockwise.copy()
    ccw, reversed_flag = ensure_ccw(clockwise)
    assert reversed_flag
    assert is_ccw(ccw)
    assert np.array_equal(clockwise, original)


def test_resample_preserves_length_and_start_point() -> None:
    circle = make_circle_track(count=360).boundaries.outer
    points = as_array(circle.points)
    original_length = closed_polyline_length(points)
    resampled = resample_closed_polyline(points, 2048, start_point=(60.0, 0.0))
    assert resampled[0] == pytest.approx((60.0, 0.0), abs=1e-12)
    # Uniform resampling cuts corners at original vertices, so the resampled
    # polygon is slightly shorter than the original polygon.
    assert closed_polyline_length(resampled) == pytest.approx(original_length, rel=1e-4)


def test_circle_centerline_length_width_and_curvature() -> None:
    summary = summarize_track(make_circle_track(), count=2048)
    assert summary.lap_length_m == pytest.approx(100.0 * math.pi, rel=5e-5)
    assert summary.min_width_m == pytest.approx(20.0, abs=1e-3)
    assert summary.max_width == pytest.approx(20.0, abs=1e-3)
    median_curvature = float(np.median(summary.curvature))
    assert median_curvature == pytest.approx(1.0 / 50.0, rel=1e-3)


@pytest.mark.parametrize(
    ("track", "expected_length", "length_rel_tol", "width_tol"),
    [
        (make_circle_track(count=360), 100.0 * math.pi, 5e-5, 1e-3),
        (make_circle_track(count=720), 100.0 * math.pi, 5e-5, 1e-3),
        (make_oval_track(count=720), 200.0 + 60.0 * math.pi, 1e-4, 1e-2),
        (make_oval_track(count=1440), 200.0 + 60.0 * math.pi, 1e-4, 1e-2),
    ],
)
def test_analytic_geometry_at_supported_sampling_densities(
    track: Track,
    expected_length: float,
    length_rel_tol: float,
    width_tol: float,
) -> None:
    summary = summarize_track(track)
    assert summary.lap_length_m == pytest.approx(expected_length, rel=length_rel_tol)
    assert summary.min_width_m == pytest.approx(20.0, abs=width_tol)
    assert summary.max_width == pytest.approx(20.0, abs=width_tol)


def test_oval_centerline_length_width_and_curvature() -> None:
    summary = summarize_track(make_oval_track(), count=2048)
    expected_length = 200.0 + 60.0 * math.pi
    assert summary.lap_length_m == pytest.approx(expected_length, rel=1e-4)
    assert summary.min_width_m == pytest.approx(20.0, abs=1e-2)
    assert summary.max_width == pytest.approx(20.0, abs=1e-2)
    curvature = np.abs(summary.curvature)
    assert float(np.percentile(curvature, 90)) == pytest.approx(1.0 / 30.0, rel=0.02)
    assert float(np.percentile(curvature, 10)) < 0.01


def test_centerline_uses_start_finish_boundary_points() -> None:
    result = derive_centerline(make_circle_track(), count=512)
    assert result.outer_start == pytest.approx((60.0, 0.0), abs=1e-9)
    assert result.inner_start == pytest.approx((40.0, 0.0), abs=1e-9)
    assert result.centerline[0] == pytest.approx((50.0, 0.0), abs=1e-9)


def test_compute_curvature_rejects_short_input() -> None:
    with pytest.raises(ValueError):
        compute_curvature(np.asarray([(0.0, 0.0), (1.0, 0.0)]))
