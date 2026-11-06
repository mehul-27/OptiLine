from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString

from optiline.core.planning import (
    BoundaryOffsetError,
    baseline_fingerprint,
    build_offset_bounds,
    build_racing_line_path,
    evaluate_periodic_offsets,
)
from optiline.core.planning.racing_line import _intersection_points


def test_zero_offsets_reproduce_m41_centerline(
    circle_track,
    circle_path,
) -> None:
    path = build_racing_line_path(
        circle_track,
        circle_path,
        tuple(0.0 for _ in range(16)),
        0.5,
    )
    assert path.unique_sample_count == circle_path.unique_sample_count
    assert len(path.samples) == len(circle_path.samples)
    assert path.length_m == pytest.approx(circle_path.length_m)
    assert path.reference_baseline_id == baseline_fingerprint(circle_path)
    assert all(offset == pytest.approx(0.0) for offset in path.offset_samples_m)
    for candidate, baseline in zip(path.samples, circle_path.samples, strict=True):
        assert candidate.x_m == pytest.approx(baseline.x_m)
        assert candidate.y_m == pytest.approx(baseline.y_m)
        assert candidate.s_m == pytest.approx(baseline.s_m)
        assert candidate.curvature_1_per_m == pytest.approx(baseline.curvature_1_per_m)


def test_circle_offset_bounds_have_positive_margin(circle_track, circle_path) -> None:
    bounds = build_offset_bounds(circle_track, circle_path, 0.5)
    assert len(bounds.lower_m) == len(circle_path.samples)
    assert len(bounds.upper_m) == len(circle_path.samples)
    assert bounds.margin_m == pytest.approx(0.5)
    assert bounds.lower_m[0] == pytest.approx(-9.5, abs=0.01)
    assert bounds.upper_m[0] == pytest.approx(9.5, abs=0.01)
    assert all(
        lower <= 0.0 <= upper for lower, upper in zip(bounds.lower_m, bounds.upper_m, strict=True)
    )


def test_narrow_corridor_with_margin_is_infeasible(narrow_track, narrow_path) -> None:
    with pytest.raises(BoundaryOffsetError, match="no feasible offset interval"):
        build_offset_bounds(narrow_track, narrow_path, 1.5)


def test_boundary_overlap_is_rejected_as_ambiguous() -> None:
    with pytest.raises(BoundaryOffsetError, match="overlaps"):
        _intersection_points(LineString([(0.0, 0.0), (1.0, 0.0)]))


def test_periodic_offsets_close_and_cross_zero(circle_path) -> None:
    controls = (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0)
    s_values = np.asarray([sample.s_m for sample in circle_path.samples[:-1]])
    offsets = evaluate_periodic_offsets(controls, circle_path.length_m, s_values)
    assert np.all(np.isfinite(offsets))
    starts = evaluate_periodic_offsets(
        controls,
        circle_path.length_m,
        np.asarray([0.0]),
    )
    ends = evaluate_periodic_offsets(
        controls,
        circle_path.length_m,
        np.asarray([circle_path.length_m]),
    )
    assert starts[0] == pytest.approx(ends[0])
