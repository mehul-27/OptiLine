"""M4.1 baseline path construction from the validated track centreline."""

from __future__ import annotations

import numpy as np

from optiline.core.planning.config import PLANNING_SCHEMA_VERSION
from optiline.core.planning.contracts import BaselinePath, PathSample
from optiline.core.track.geometry import compute_curvature
from optiline.core.track.schema import Track
from optiline.core.track.validation import validate_track


def build_baseline_path(track: Track, *, sample_count: int = 1024) -> BaselinePath:
    """Build a closed path from the validated track centreline.

    The centreline is a geometric baseline, not a racing line and not a
    substitute for corridor validity. The track must be simulation-ready under
    the M1.1 track validation contract. The returned path uses
    ``sample_count + 1`` samples: the first geometric point is repeated at the
    end at arc length ``length_m`` so the final interval is explicit.
    """

    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 4:
        raise ValueError("sample_count must be an integer of at least four")

    report = validate_track(track, count=sample_count)
    if not report.simulation_ready or report.summary is None:
        codes = sorted({issue.code for issue in report.issues if issue.severity == "error"})
        detail = ", ".join(codes) if codes else "track validation failed"
        raise ValueError(f"track is not simulation-ready: {detail}")

    unique = np.asarray(report.summary.centerline, dtype=np.float64)
    segment_vectors = np.roll(unique, -1, axis=0) - unique
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    if np.any(segment_lengths <= 1e-12):
        raise ValueError("track centreline contains a zero-length segment")

    unit_tangents = segment_vectors / segment_lengths[:, None]
    s_values = np.concatenate((np.zeros(1), np.cumsum(segment_lengths)))
    length_m = float(s_values[-1])
    positions = np.vstack((unique, unique[:1]))
    tangents = np.vstack((unit_tangents, unit_tangents[:1]))
    curvature = compute_curvature(unique)
    curvature_closed = np.concatenate((curvature, curvature[:1]))

    samples = tuple(
        PathSample(
            x_m=float(positions[index, 0]),
            y_m=float(positions[index, 1]),
            s_m=float(s_values[index]),
            tangent_x=float(tangents[index, 0]),
            tangent_y=float(tangents[index, 1]),
            curvature_1_per_m=float(curvature_closed[index]),
        )
        for index in range(len(positions))
    )
    return BaselinePath(
        schema_version=PLANNING_SCHEMA_VERSION,
        source_track_id=track.track_id,
        source_track_name=track.name,
        length_m=length_m,
        unique_sample_count=len(unique),
        closed=True,
        samples=samples,
    )
