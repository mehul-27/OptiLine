from __future__ import annotations

import json
from pathlib import Path

import pytest

from optiline.core.track.io import (
    TrackLoadError,
    load_track,
    save_track,
    track_from_json,
    track_to_json,
)
from tests.fixtures.track_builders import (
    make_circle_track,
    make_oval_track,
    make_uncalibrated_track,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "track"


def test_json_round_trip() -> None:
    track = make_circle_track(count=64)
    loaded = track_from_json(track_to_json(track))
    assert loaded == track


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    track = make_oval_track(count=128)
    path = tmp_path / "oval.json"
    save_track(track, path)
    assert load_track(path) == track


def test_checked_in_circle_fixture_loads() -> None:
    track = load_track(FIXTURES / "circle_r50.json")
    assert track.track_id == "circle-r50"
    assert track.units == "m"


def test_checked_in_oval_fixture_loads() -> None:
    track = load_track(FIXTURES / "oval_100x60.json")
    assert track.track_id == "oval-100x60"
    assert track.units == "m"


def test_unknown_field_is_rejected() -> None:
    data = make_circle_track(count=32).model_dump(mode="json")
    data["unexpected"] = True
    with pytest.raises(TrackLoadError) as error:
        track_from_json(json.dumps(data))
    assert error.value.code == "SCHEMA_INVALID"


def test_unsupported_schema_version_is_rejected() -> None:
    data = make_circle_track(count=32).model_dump(mode="json")
    data["schema_version"] = 2
    with pytest.raises(TrackLoadError) as error:
        track_from_json(json.dumps(data))
    assert error.value.code == "SCHEMA_VERSION_UNSUPPORTED"


def test_non_integer_schema_version_is_rejected() -> None:
    data = make_circle_track(count=32).model_dump(mode="json")
    data["schema_version"] = 1.0
    with pytest.raises(TrackLoadError) as error:
        track_from_json(json.dumps(data))
    assert error.value.code == "SCHEMA_INVALID"


def test_non_finite_json_number_is_rejected() -> None:
    data = make_circle_track(count=32).model_dump(mode="json")
    data["boundaries"]["outer"]["points"][0][0] = float("nan")
    payload = json.dumps(data, allow_nan=True)
    with pytest.raises(TrackLoadError) as error:
        track_from_json(payload)
    assert error.value.code == "SCHEMA_INVALID"


def test_non_object_json_is_rejected() -> None:
    with pytest.raises(TrackLoadError) as error:
        track_from_json("[]")
    assert error.value.code == "SCHEMA_INVALID"


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(TrackLoadError) as error:
        track_from_json("{not json")
    assert error.value.code == "JSON_INVALID"


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(TrackLoadError) as error:
        load_track(tmp_path / "missing.json")
    assert error.value.code == "IO_ERROR"


def test_uncalibrated_track_round_trips() -> None:
    track = make_uncalibrated_track()
    loaded = track_from_json(track_to_json(track))
    assert loaded == track
    assert loaded.units == "unspecified"
    assert loaded.calibration.status == "uncalibrated"


def test_derived_values_are_not_persisted() -> None:
    text = track_to_json(make_circle_track(count=32))
    assert "lap_length_m" not in text
    assert "centerline" not in text
    assert "validation" not in text
