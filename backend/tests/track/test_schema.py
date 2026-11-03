from __future__ import annotations

import pytest
from pydantic import ValidationError

from optiline.core.track.schema import (
    SCHEMA_VERSION,
    Boundary,
    Calibration,
    StartFinish,
    Track,
)
from tests.fixtures.track_builders import make_circle_track, make_uncalibrated_track


def test_valid_track_parses() -> None:
    track = make_circle_track()
    assert track.schema_version == SCHEMA_VERSION
    assert track.units == "m"
    assert len(track.boundaries.outer.points) == 360
    assert len(track.boundaries.inner.points) == 360
    assert track.calibration.status == "calibrated"


def test_missing_schema_version_fails() -> None:
    data = make_circle_track().model_dump(mode="json")
    del data["schema_version"]
    with pytest.raises(ValidationError):
        Track.model_validate(data)


def test_unknown_field_fails() -> None:
    data = make_circle_track().model_dump(mode="json")
    data["unexpected"] = 1
    with pytest.raises(ValidationError):
        Track.model_validate(data)


def test_unsupported_schema_version_fails() -> None:
    data = make_circle_track().model_dump(mode="json")
    data["schema_version"] = 999
    with pytest.raises(ValidationError):
        Track.model_validate(data)


def test_calibrated_track_requires_metres() -> None:
    data = make_circle_track().model_dump(mode="json")
    data["units"] = "unspecified"
    with pytest.raises(ValidationError):
        Track.model_validate(data)


def test_uncalibrated_track_requires_unspecified_units() -> None:
    data = make_uncalibrated_track().model_dump(mode="json")
    data["units"] = "m"
    with pytest.raises(ValidationError):
        Track.model_validate(data)


def test_boundary_requires_three_points() -> None:
    with pytest.raises(ValidationError):
        Boundary(points=((0.0, 0.0), (1.0, 0.0)))


def test_boundary_rejects_repeated_closing_point() -> None:
    with pytest.raises(ValidationError):
        Boundary(points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)))


def test_boundary_rejects_duplicate_consecutive_points() -> None:
    with pytest.raises(ValidationError):
        Boundary(points=((0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 1.0)))


def test_boundary_rejects_non_finite_points() -> None:
    with pytest.raises(ValidationError):
        Boundary(points=((0.0, 0.0), (1.0, float("nan")), (1.0, 1.0)))


def test_start_finish_forward_must_be_unit_length() -> None:
    with pytest.raises(ValidationError):
        StartFinish(line=((0.0, 0.0), (1.0, 0.0)), forward=(0.0, 2.0))


def test_start_finish_endpoints_must_differ() -> None:
    with pytest.raises(ValidationError):
        StartFinish(line=((0.0, 0.0), (0.0, 0.0)), forward=(0.0, 1.0))


def test_uncalibrated_calibration_rejects_transform_values() -> None:
    with pytest.raises(ValidationError):
        Calibration(status="uncalibrated", scale_m_per_unit=1.0)
