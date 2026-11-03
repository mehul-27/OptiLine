"""Track schema, geometry, calibration, validation and persistence."""

from optiline.core.track.calibration import (
    SimilarityTransform,
    calibrate_from_known_distance,
    calibrate_track_from_known_distance,
    uncalibrated_calibration,
)
from optiline.core.track.geometry import (
    CenterlineResult,
    TrackSummary,
    closed_polyline_length,
    compute_curvature,
    derive_centerline,
    ensure_ccw,
    is_ccw,
    signed_area,
    summarize_track,
)
from optiline.core.track.io import (
    TrackLoadError,
    load_track,
    save_track,
    track_from_json,
    track_to_json,
)
from optiline.core.track.schema import (
    SCHEMA_VERSION,
    Boundary,
    Calibration,
    StartFinish,
    Track,
    TrackBoundaries,
)
from optiline.core.track.validation import (
    TrackValidationReport,
    ValidationIssue,
    normalize_track,
    validate_track,
)

__all__ = [
    "SCHEMA_VERSION",
    "Boundary",
    "Calibration",
    "CenterlineResult",
    "SimilarityTransform",
    "StartFinish",
    "Track",
    "TrackBoundaries",
    "TrackLoadError",
    "TrackSummary",
    "TrackValidationReport",
    "ValidationIssue",
    "calibrate_from_known_distance",
    "calibrate_track_from_known_distance",
    "closed_polyline_length",
    "compute_curvature",
    "derive_centerline",
    "ensure_ccw",
    "is_ccw",
    "load_track",
    "normalize_track",
    "save_track",
    "signed_area",
    "summarize_track",
    "track_from_json",
    "track_to_json",
    "uncalibrated_calibration",
    "validate_track",
]
