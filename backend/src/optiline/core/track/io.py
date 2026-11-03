"""JSON persistence for the versioned track schema."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from optiline.core.track.schema import SCHEMA_VERSION, Track


class TrackLoadError(ValueError):
    """A structured track-loading failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def track_from_json(text: str) -> Track:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TrackLoadError("JSON_INVALID", f"invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise TrackLoadError("SCHEMA_INVALID", "track document must be a JSON object")

    version = data.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise TrackLoadError("SCHEMA_INVALID", "schema_version must be an integer")
    if version != SCHEMA_VERSION:
        raise TrackLoadError(
            "SCHEMA_VERSION_UNSUPPORTED",
            f"unsupported schema version {version}; expected {SCHEMA_VERSION}",
        )

    try:
        return Track.model_validate(data)
    except ValidationError as exc:
        raise TrackLoadError("SCHEMA_INVALID", str(exc)) from exc


def track_to_json(track: Track) -> str:
    return track.model_dump_json(indent=2) + "\n"


def load_track(path: str | Path) -> Track:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise TrackLoadError("IO_ERROR", f"could not read track: {exc}") from exc
    return track_from_json(text)


def save_track(track: Track, path: str | Path) -> None:
    try:
        Path(path).write_text(track_to_json(track), encoding="utf-8")
    except OSError as exc:
        raise TrackLoadError("IO_ERROR", f"could not write track: {exc}") from exc
