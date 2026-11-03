from __future__ import annotations

import math

import numpy as np
import pytest

from optiline.core.track.calibration import (
    SimilarityTransform,
    calibrate_from_known_distance,
    calibrate_track_from_known_distance,
)
from optiline.core.track.geometry import summarize_track
from tests.fixtures.track_builders import make_circle_track, make_uncalibrated_track


def test_known_distance_maps_start_to_origin_and_end_to_x_axis() -> None:
    transform = calibrate_from_known_distance((0.0, 0.0), (100.0, 0.0), 50.0)
    transformed = transform.apply(np.asarray([(0.0, 0.0), (100.0, 0.0)]))
    assert transform.scale == pytest.approx(0.5)
    assert transform.rotation_rad == pytest.approx(0.0)
    assert transformed[0] == pytest.approx((0.0, 0.0))
    assert transformed[1] == pytest.approx((50.0, 0.0))


def test_vertical_reference_segment_rotates_to_x_axis() -> None:
    transform = calibrate_from_known_distance((0.0, 0.0), (0.0, 100.0), 200.0)
    transformed = transform.apply(np.asarray([(0.0, 0.0), (0.0, 100.0)]))
    assert transform.scale == pytest.approx(2.0)
    assert transform.rotation_rad == pytest.approx(-math.pi / 2.0)
    assert transformed[0] == pytest.approx((0.0, 0.0))
    assert transformed[1] == pytest.approx((200.0, 0.0))


def test_inverse_round_trip() -> None:
    transform = calibrate_from_known_distance((10.0, 20.0), (40.0, 60.0), 100.0)
    points = np.asarray([(10.0, 20.0), (25.0, 35.0), (40.0, 60.0)])
    transformed = transform.apply(points)
    restored = transform.inverse(transformed)
    assert np.allclose(restored, points, atol=1e-12)


def test_invalid_known_distance_inputs() -> None:
    with pytest.raises(ValueError):
        calibrate_from_known_distance((0.0, 0.0), (1.0, 0.0), 0.0)
    with pytest.raises(ValueError):
        calibrate_from_known_distance((0.0, 0.0), (1.0, 0.0), -1.0)
    with pytest.raises(ValueError):
        calibrate_from_known_distance((0.0, 0.0), (0.0, 0.0), 10.0)
    with pytest.raises(ValueError):
        calibrate_from_known_distance((0.0, 0.0), (1.0, 0.0), math.nan)
    with pytest.raises(ValueError):
        calibrate_from_known_distance((0.0, 0.0), (1.0, 0.0), math.inf)


@pytest.mark.parametrize(
    ("source_p1", "source_p2"),
    [
        ((math.nan, 0.0), (1.0, 0.0)),
        ((0.0, 0.0), (math.inf, 0.0)),
        ((0.0, 0.0), (0.0, math.nan)),
    ],
)
def test_non_finite_source_points_are_rejected(
    source_p1: tuple[float, float],
    source_p2: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="finite"):
        calibrate_from_known_distance(source_p1, source_p2, 1.0)


@pytest.mark.parametrize("scale", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_similarity_transform_rejects_invalid_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        SimilarityTransform(scale=scale, rotation_rad=0.0, translation=(0.0, 0.0))


@pytest.mark.parametrize(
    ("rotation_rad", "translation"),
    [
        (math.nan, (0.0, 0.0)),
        (math.inf, (0.0, 0.0)),
        (0.0, (math.inf, 0.0)),
        (0.0, (0.0, math.nan)),
    ],
)
def test_similarity_transform_rejects_non_finite_values(
    rotation_rad: float,
    translation: tuple[float, float],
) -> None:
    with pytest.raises(ValueError):
        SimilarityTransform(scale=1.0, rotation_rad=rotation_rad, translation=translation)


def test_inverse_of_valid_transform_is_finite() -> None:
    transform = SimilarityTransform(
        scale=2.0,
        rotation_rad=math.pi / 3.0,
        translation=(1.0, -2.0),
    )
    points = np.asarray([(3.0, 4.0), (-5.0, 6.0)])
    restored = transform.inverse(transform.apply(points))
    assert np.all(np.isfinite(restored))
    assert np.allclose(restored, points, atol=1e-12)


def test_calibrate_track_rejects_non_finite_source_points() -> None:
    with pytest.raises(ValueError, match="finite"):
        calibrate_track_from_known_distance(
            make_uncalibrated_track(),
            (math.nan, 0.0),
            (1.0, 0.0),
            1.0,
        )


def test_calibrate_track_scales_geometry() -> None:
    track = make_uncalibrated_track()
    calibrated = calibrate_track_from_known_distance(
        track,
        (0.0, 0.0),
        (100.0, 0.0),
        50.0,
        source="test",
    )
    assert calibrated.calibration.status == "calibrated"
    assert calibrated.units == "m"
    start, end = calibrated.start_finish.line
    assert start == pytest.approx((20.0, 0.0))
    assert end == pytest.approx((30.0, 0.0))
    assert calibrated.calibration.scale_m_per_unit == pytest.approx(0.5)


def test_calibrate_already_calibrated_track_fails() -> None:
    track = make_circle_track()
    with pytest.raises(ValueError):
        calibrate_track_from_known_distance(track, (0.0, 0.0), (1.0, 0.0), 1.0)


def test_calibration_scales_derived_track_length() -> None:
    track = make_uncalibrated_track()
    before = summarize_track(track, count=512).length_world_units
    calibrated = calibrate_track_from_known_distance(
        track,
        (0.0, 0.0),
        (100.0, 0.0),
        50.0,
        source="test",
    )
    after = summarize_track(calibrated, count=512)
    assert after.lap_length_m == pytest.approx(before * 0.5, rel=1e-9)


def test_overflowing_reference_segment_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite and distinct"):
        calibrate_from_known_distance((-1e308, 0.0), (1e308, 0.0), 1.0)


def test_underflowing_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="scale"):
        calibrate_from_known_distance((0.0, 0.0), (1e308, 0.0), 1e-308)
