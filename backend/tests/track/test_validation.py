from __future__ import annotations

import math

import numpy as np
import pytest

from optiline.core.track.geometry import as_array
from optiline.core.track.schema import Boundary, StartFinish, Track, TrackBoundaries
from optiline.core.track.validation import TrackValidationReport, validate_track
from tests.fixtures.track_builders import (
    make_bad_direction_track,
    make_circle_track,
    make_inner_outside_track,
    make_narrow_track,
    make_oval_track,
    make_self_intersecting_start_finish_track,
    make_self_intersecting_track,
    make_uncalibrated_track,
)


def _codes(report: TrackValidationReport) -> set[str]:
    return {issue.code for issue in report.issues}


def _u_shaped_track() -> Track:
    track = make_circle_track(count=128)
    outer = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    inner = [
        (2.0, 2.0),
        (8.0, 2.0),
        (8.0, 8.0),
        (6.0, 8.0),
        (6.0, 4.0),
        (4.0, 4.0),
        (4.0, 8.0),
        (2.0, 8.0),
    ]
    return track.model_copy(
        update={
            "track_id": "u-shape",
            "name": "U-shape",
            "boundaries": TrackBoundaries(
                outer=Boundary(points=outer),
                inner=Boundary(points=inner),
            ),
            "start_finish": StartFinish(
                line=((5.0, 2.0), (5.0, 0.0)),
                forward=(1.0, 0.0),
            ),
        },
    )


def test_valid_circle_is_simulation_ready() -> None:
    report = validate_track(make_circle_track(), count=1024, min_width_m=1.0)
    assert report.is_valid
    assert report.geometry_valid
    assert report.simulation_ready
    assert report.summary is not None
    assert report.summary.lap_length_m == pytest.approx(100.0 * math.pi, rel=5e-5)
    assert report.summary.min_width_m == pytest.approx(20.0, abs=1e-3)
    assert not any(issue.severity == "error" for issue in report.issues)


def test_valid_oval_is_simulation_ready() -> None:
    report = validate_track(make_oval_track(), count=1024, min_width_m=1.0)
    assert report.simulation_ready
    assert report.summary is not None
    assert report.summary.lap_length_m == pytest.approx(200.0 + 60.0 * math.pi, rel=1e-4)


def test_uncalibrated_track_is_geometrically_valid_but_not_simulation_ready() -> None:
    report = validate_track(make_uncalibrated_track(), count=512)
    assert report.geometry_valid
    assert not report.is_valid
    assert not report.simulation_ready
    assert "CALIBRATION_MISSING" in _codes(report)
    assert report.summary is not None
    assert report.summary.lap_length_m is None
    assert report.summary.min_width_m is None


def test_self_intersecting_boundary_is_rejected() -> None:
    report = validate_track(make_self_intersecting_track(), count=256)
    assert not report.geometry_valid
    assert _codes(report) & {"BOUNDARY_SELF_INTERSECTION", "CORRIDOR_INVALID"}


def test_inner_outside_outer_is_rejected() -> None:
    report = validate_track(make_inner_outside_track(), count=256)
    assert not report.geometry_valid
    assert "INNER_NOT_INSIDE_OUTER" in _codes(report)


def test_narrow_corridor_is_rejected() -> None:
    report = validate_track(make_narrow_track(), count=512, min_width_m=1.0)
    assert "CORRIDOR_TOO_NARROW" in _codes(report)
    assert not report.simulation_ready


def test_start_finish_must_touch_boundaries() -> None:
    report = validate_track(make_self_intersecting_start_finish_track(), count=512)
    assert "START_FINISH_ENDPOINT_NOT_ON_BOUNDARY" in _codes(report)


def test_start_finish_direction_must_be_across_track() -> None:
    report = validate_track(make_bad_direction_track(), count=512)
    assert "START_FINISH_DIRECTION_INVALID" in _codes(report)


def test_orientation_normalization_warns_but_preserves_validity() -> None:
    track = make_circle_track(count=256)
    outer = as_array(track.boundaries.outer.points)[::-1]
    inner = as_array(track.boundaries.inner.points)[::-1]
    reversed_track = track.model_copy(
        update={
            "boundaries": TrackBoundaries(
                outer=Boundary(points=[(float(x), float(y)) for x, y in outer]),
                inner=Boundary(points=[(float(x), float(y)) for x, y in inner]),
            ),
        },
    )
    report = validate_track(reversed_track, count=512)
    assert report.geometry_valid
    warnings = [issue for issue in report.issues if issue.code == "BOUNDARY_ORIENTATION_NORMALIZED"]
    assert len(warnings) == 2


def test_non_convex_centerline_discontinuity_is_not_simulation_ready() -> None:
    report = validate_track(_u_shaped_track(), count=512)
    assert report.geometry_valid
    assert not report.is_valid
    assert not report.simulation_ready
    assert "CENTERLINE_DISCONTINUITY" in _codes(report)


def _rectangular_track() -> Track:
    track = make_circle_track(count=128)
    outer = [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)]
    inner = [(1.0, 1.0), (9.0, 1.0), (9.0, 3.0), (1.0, 3.0)]
    return track.model_copy(
        update={
            "track_id": "rectangle",
            "name": "Rectangle",
            "boundaries": TrackBoundaries(
                outer=Boundary(points=outer),
                inner=Boundary(points=inner),
            ),
            "start_finish": StartFinish(
                line=((5.0, 1.0), (5.0, 0.0)),
                forward=(1.0, 0.0),
            ),
        },
    )


def test_sharp_corner_rectangle_has_no_centerline_discontinuity() -> None:
    report = validate_track(_rectangular_track(), count=512)
    assert report.simulation_ready
    assert "CENTERLINE_DISCONTINUITY" not in _codes(report)


def test_clockwise_travel_direction_is_supported() -> None:
    track = make_circle_track(count=360)
    clockwise = track.model_copy(
        update={
            "start_finish": StartFinish(
                line=track.start_finish.line,
                forward=(0.0, -1.0),
            ),
        },
    )
    report = validate_track(clockwise, count=512)
    assert report.simulation_ready
    assert "CENTERLINE_DISCONTINUITY" not in _codes(report)
    assert report.summary is not None
    assert report.summary.lap_length_m == pytest.approx(100.0 * math.pi, rel=5e-5)


def test_near_duplicate_boundary_point_is_detected() -> None:
    track = make_circle_track(count=128)
    outer = list(track.boundaries.outer.points)
    point = outer[10]
    outer.insert(11, (point[0] + 1e-13, point[1]))
    broken = track.model_copy(
        update={
            "boundaries": TrackBoundaries(
                outer=Boundary(points=outer),
                inner=track.boundaries.inner,
            ),
        },
    )
    report = validate_track(broken, count=512)
    assert not report.geometry_valid
    assert not report.simulation_ready
    assert "CORRIDOR_INVALID" in _codes(report)


def test_validation_is_deterministic() -> None:
    track = make_circle_track(count=256)
    first = validate_track(track, count=512)
    second = validate_track(track, count=512)
    assert [issue.code for issue in first.issues] == [issue.code for issue in second.issues]
    assert first.summary is not None
    assert second.summary is not None
    assert first.summary.lap_length_m == pytest.approx(second.summary.lap_length_m)
    assert first.summary.min_width_m == pytest.approx(second.summary.min_width_m)
    assert np.array_equal(first.summary.centerline, second.summary.centerline)
