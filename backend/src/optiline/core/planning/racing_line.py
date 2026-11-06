"""Racing-line path, periodic offsets and corridor offset bounds for M4.2."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from shapely.geometry import LineString
from shapely.ops import unary_union

from optiline.core.planning.config import PLANNING_SCHEMA_VERSION
from optiline.core.planning.contracts import BaselinePath, PathSample
from optiline.core.track.geometry import as_array, compute_curvature
from optiline.core.track.schema import Track
from optiline.core.track.validation import normalize_track

FloatArray = npt.NDArray[np.float64]
_CLOSURE_TOL = 1e-9
_UNIT_TOL = 1e-9


class BoundaryOffsetError(ValueError):
    """Raised when normal-based corridor offset bounds are invalid or ambiguous."""


@dataclass(frozen=True)
class RacingLineOptimiserMetadata:
    """Provenance for the constrained geometric optimiser."""

    method: str
    scipy_version: str
    success: bool
    status: int
    message: str
    iterations: int
    objective_value: float
    control_point_count: int
    boundary_margin_m: float
    max_iterations: int
    ftol: float
    lap_time_improvement_threshold_s: float
    start_strategy: str


@dataclass(frozen=True)
class OffsetBounds:
    """Feasible lateral offsets along the reference path normals."""

    lower_m: tuple[float, ...]
    upper_m: tuple[float, ...]
    margin_m: float

    def __post_init__(self) -> None:
        if len(self.lower_m) < 4 or len(self.lower_m) != len(self.upper_m):
            raise ValueError("offset bounds must have matching arrays of at least four")
        if not math.isfinite(self.margin_m) or self.margin_m <= 0.0:
            raise ValueError("margin_m must be finite and positive")
        for lower, upper in zip(self.lower_m, self.upper_m, strict=True):
            if not (math.isfinite(lower) and math.isfinite(upper)):
                raise ValueError("offset bounds must be finite")
            if lower >= upper:
                raise ValueError("offset lower bound must be below upper bound")
            if lower > 0.0 or upper < 0.0:
                raise ValueError("centerline offset zero must remain feasible")


@dataclass(frozen=True)
class RacingLinePath:
    """Closed racing-line path with explicit periodic offset provenance."""

    schema_version: int
    source_track_id: str
    source_track_name: str
    reference_baseline_id: str
    length_m: float
    unique_sample_count: int
    closed: bool
    samples: tuple[PathSample, ...]
    offset_samples_m: tuple[float, ...]
    control_offsets_m: tuple[float, ...]
    boundary_margin_m: float
    optimiser_metadata: RacingLineOptimiserMetadata | None = None

    def __post_init__(self) -> None:
        if self.schema_version != PLANNING_SCHEMA_VERSION:
            raise ValueError(f"unsupported racing-line schema_version: {self.schema_version}")
        if not (self.source_track_id and self.source_track_name and self.reference_baseline_id):
            raise ValueError("racing-line identity fields must not be empty")
        if not math.isfinite(self.length_m) or self.length_m <= 0.0:
            raise ValueError("length_m must be positive and finite")
        if self.unique_sample_count < 3:
            raise ValueError("unique_sample_count must be at least three")
        if not self.closed:
            raise ValueError("M4.2 requires a closed racing line")
        if len(self.samples) != self.unique_sample_count + 1:
            raise ValueError("racing line must contain unique_sample_count + 1 samples")
        if len(self.offset_samples_m) != self.unique_sample_count + 1:
            raise ValueError("offsets must contain unique_sample_count + 1 samples")
        if len(self.control_offsets_m) < 4:
            raise ValueError("racing line needs at least four spline control offsets")
        if not math.isfinite(self.boundary_margin_m) or self.boundary_margin_m <= 0.0:
            raise ValueError("boundary_margin_m must be finite and positive")
        for value in (*self.offset_samples_m, *self.control_offsets_m):
            if not math.isfinite(value):
                raise ValueError("racing-line offsets must be finite")

        first_path = self.samples[0]
        last_path = self.samples[-1]
        if (
            abs(first_path.x_m - last_path.x_m) > _CLOSURE_TOL
            or abs(first_path.y_m - last_path.y_m) > _CLOSURE_TOL
        ):
            raise ValueError("racing-line start and end positions must match")
        if abs(first_path.s_m) > _CLOSURE_TOL:
            raise ValueError("racing line must start at s=0")
        if abs(last_path.s_m - self.length_m) > _CLOSURE_TOL:
            raise ValueError("racing-line final sample must be at length_m")
        if abs(self.offset_samples_m[0] - self.offset_samples_m[-1]) > _CLOSURE_TOL:
            raise ValueError("racing-line start and end offsets must match")
        previous = 0.0
        for sample in self.samples:
            if sample.s_m <= previous and sample is not first_path:
                raise ValueError("racing-line arc length must increase strictly")
            previous = sample.s_m
            tangent_norm = math.hypot(sample.tangent_x, sample.tangent_y)
            if abs(tangent_norm - 1.0) > _UNIT_TOL:
                raise ValueError("racing-line tangent must be a unit vector")

    def as_baseline_path(self) -> BaselinePath:
        """Adapt this path to the M4.1 path interface without changing M4.1."""

        return BaselinePath(
            schema_version=PLANNING_SCHEMA_VERSION,
            source_track_id=self.source_track_id,
            source_track_name=self.source_track_name,
            length_m=self.length_m,
            unique_sample_count=self.unique_sample_count,
            closed=self.closed,
            samples=self.samples,
        )


def baseline_fingerprint(path: BaselinePath) -> str:
    """Return a deterministic content identity for the M4.1 reference path."""

    digest = hashlib.sha256()
    digest.update(path.source_track_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(format(path.length_m, ".17g").encode("ascii"))
    digest.update(b"\0")
    digest.update(str(path.unique_sample_count).encode("ascii"))
    for sample in path.samples:
        for value in (
            sample.x_m,
            sample.y_m,
            sample.s_m,
            sample.tangent_x,
            sample.tangent_y,
            sample.curvature_1_per_m,
        ):
            digest.update(format(value, ".17g").encode("ascii"))
            digest.update(b",")
        digest.update(b"\n")
    return digest.hexdigest()


def _periodic_spline(control_offsets: FloatArray, length_m: float) -> CubicSpline:
    if len(control_offsets) < 4:
        raise ValueError("at least four control offsets are required")
    knots = np.linspace(0.0, length_m, len(control_offsets) + 1)
    values = np.concatenate((control_offsets, control_offsets[:1]))
    return CubicSpline(knots, values, bc_type="periodic", extrapolate="periodic")


def evaluate_periodic_offsets(
    control_offsets: tuple[float, ...],
    length_m: float,
    s_values: FloatArray,
) -> FloatArray:
    """Evaluate the periodic offset spline at supplied reference arc lengths."""

    control = np.asarray(control_offsets, dtype=np.float64)
    if len(control) < 4 or np.any(~np.isfinite(control)):
        raise ValueError("control offsets must be finite and contain at least four values")
    if not math.isfinite(length_m) or length_m <= 0.0:
        raise ValueError("length_m must be finite and positive")
    spline = _periodic_spline(control, length_m)
    return np.asarray(spline(np.mod(s_values, length_m)), dtype=np.float64)


def _intersection_points(geometry: object) -> list[tuple[float, float]]:
    if geometry.is_empty:  # type: ignore[attr-defined]
        return []
    geometry_type = geometry.geom_type  # type: ignore[attr-defined]
    if geometry_type == "Point":
        coordinates = geometry.coords[0]  # type: ignore[attr-defined]
        return [(float(coordinates[0]), float(coordinates[1]))]
    if geometry_type in ("MultiPoint", "GeometryCollection"):
        points: list[tuple[float, float]] = []
        for part in geometry.geoms:  # type: ignore[attr-defined]
            points.extend(_intersection_points(part))
        return points
    if geometry_type in ("LineString", "MultiLineString"):
        raise BoundaryOffsetError("normal line overlaps a boundary segment")
    raise BoundaryOffsetError(f"unsupported normal/boundary intersection: {geometry_type}")


def build_offset_bounds(
    track: Track,
    baseline_path: BaselinePath,
    margin_m: float,
) -> OffsetBounds:
    """Derive one unambiguous feasible offset interval per reference sample.

    A normal line is intersected with both track boundaries. The nearest
    boundary intersection on each side defines the local corridor; farther
    intersections are beyond the drivable interval. Multiple distinct nearest
    intersections on one side, a boundary overlap or a missing side are rejected
    as ambiguous rather than silently reduced to a guessed interval.
    """

    if not math.isfinite(margin_m) or margin_m <= 0.0:
        raise ValueError("margin_m must be finite and positive")
    if baseline_path.source_track_id != track.track_id:
        raise BoundaryOffsetError("baseline path source track does not match track")
    normalized, _ = normalize_track(track)
    outer = as_array(normalized.boundaries.outer.points)
    inner = as_array(normalized.boundaries.inner.points)
    outer_line = LineString(np.vstack((outer, outer[0])))
    inner_line = LineString(np.vstack((inner, inner[0])))
    boundary_union = unary_union([outer_line, inner_line])
    outer_bounds = outer_line.bounds
    extent = max(outer_bounds[2] - outer_bounds[0], outer_bounds[3] - outer_bounds[1], 1.0)
    reach = max(1.0, 2.0 * extent)
    tolerance = max(1e-9, 1e-7 * extent)

    unique_count = baseline_path.unique_sample_count
    lower: list[float] = []
    upper: list[float] = []
    for index in range(unique_count):
        sample = baseline_path.samples[index]
        normal = np.asarray((-sample.tangent_y, sample.tangent_x), dtype=np.float64)
        origin = np.asarray((sample.x_m, sample.y_m), dtype=np.float64)
        normal_line = LineString(
            [
                tuple(origin - reach * normal),
                tuple(origin + reach * normal),
            ],
        )
        intersection = normal_line.intersection(boundary_union)
        points = _intersection_points(intersection)
        distances = [
            float(np.dot(np.asarray(point, dtype=np.float64) - origin, normal)) for point in points
        ]
        distances = [value for value in distances if abs(value) > tolerance]
        positive = sorted(value for value in distances if value > 0.0)
        negative = sorted(-value for value in distances if value < 0.0)
        if not positive or not negative:
            raise BoundaryOffsetError(
                f"missing normal-boundary intersection at path sample {index}",
            )
        if len(positive) > 1 and positive[1] - positive[0] <= tolerance:
            raise BoundaryOffsetError(
                f"ambiguous nearest positive boundary intersections at path sample {index}",
            )
        if len(negative) > 1 and negative[1] - negative[0] <= tolerance:
            raise BoundaryOffsetError(
                f"ambiguous nearest negative boundary intersections at path sample {index}",
            )
        lower_value = -negative[0] + margin_m
        upper_value = positive[0] - margin_m
        if lower_value > 0.0 or upper_value < 0.0 or lower_value >= upper_value:
            raise BoundaryOffsetError(
                f"no feasible offset interval with margin at path sample {index}",
            )
        lower.append(lower_value)
        upper.append(upper_value)

    return OffsetBounds(
        lower_m=tuple((*lower, lower[0])),
        upper_m=tuple((*upper, upper[0])),
        margin_m=margin_m,
    )


def build_racing_line_path(
    track: Track,
    baseline_path: BaselinePath,
    control_offsets: tuple[float, ...],
    boundary_margin_m: float,
    optimiser_metadata: RacingLineOptimiserMetadata | None = None,
) -> RacingLinePath:
    """Build a closed racing line from periodic cubic offset coefficients."""

    if baseline_path.source_track_id != track.track_id:
        raise BoundaryOffsetError("baseline path source track does not match track")
    control = np.asarray(control_offsets, dtype=np.float64)
    if len(control) < 4 or np.any(~np.isfinite(control)):
        raise ValueError("control offsets must be finite and contain at least four values")
    if not math.isfinite(boundary_margin_m) or boundary_margin_m <= 0.0:
        raise ValueError("boundary_margin_m must be finite and positive")

    unique_count = baseline_path.unique_sample_count
    if len(baseline_path.samples) != unique_count + 1:
        raise ValueError("baseline path must contain unique_sample_count + 1 samples")
    reference_s = np.asarray(
        [baseline_path.samples[index].s_m for index in range(unique_count)],
        dtype=np.float64,
    )
    spline = _periodic_spline(control, baseline_path.length_m)
    offset_unique = spline(np.mod(reference_s, baseline_path.length_m))
    offset_samples = np.concatenate((offset_unique, offset_unique[:1]))

    centerline = np.asarray(
        [(sample.x_m, sample.y_m) for sample in baseline_path.samples[:-1]],
        dtype=np.float64,
    )
    tangents = np.asarray(
        [(sample.tangent_x, sample.tangent_y) for sample in baseline_path.samples[:-1]],
        dtype=np.float64,
    )
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    positions = centerline + offset_unique[:, None] * normals
    segment_vectors = np.roll(positions, -1, axis=0) - positions
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    if np.any(segment_lengths <= 1e-12) or np.any(~np.isfinite(segment_lengths)):
        raise ValueError("racing line contains a degenerate segment")
    unit_tangents = segment_vectors / segment_lengths[:, None]
    s_values = np.concatenate((np.zeros(1), np.cumsum(segment_lengths)))
    length_m = float(s_values[-1])
    positions_closed = np.vstack((positions, positions[:1]))
    tangents_closed = np.vstack((unit_tangents, unit_tangents[:1]))
    curvature = compute_curvature(positions)
    curvature_closed = np.concatenate((curvature, curvature[:1]))
    samples = tuple(
        PathSample(
            x_m=float(positions_closed[index, 0]),
            y_m=float(positions_closed[index, 1]),
            s_m=float(s_values[index]),
            tangent_x=float(tangents_closed[index, 0]),
            tangent_y=float(tangents_closed[index, 1]),
            curvature_1_per_m=float(curvature_closed[index]),
        )
        for index in range(len(positions_closed))
    )
    return RacingLinePath(
        schema_version=PLANNING_SCHEMA_VERSION,
        source_track_id=track.track_id,
        source_track_name=track.name,
        reference_baseline_id=baseline_fingerprint(baseline_path),
        length_m=length_m,
        unique_sample_count=unique_count,
        closed=True,
        samples=samples,
        offset_samples_m=tuple(float(value) for value in offset_samples),
        control_offsets_m=tuple(float(value) for value in control),
        boundary_margin_m=boundary_margin_m,
        optimiser_metadata=optimiser_metadata,
    )
