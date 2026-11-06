"""M4.1 baseline path and speed-profile planning."""

from optiline.core.planning.config import (
    ILLUSTRATIVE_LATERAL_ACCELERATION_MPS2,
    PLANNING_SCHEMA_VERSION,
    PlanningLimits,
)
from optiline.core.planning.contracts import (
    BaselinePath,
    PathSample,
    SpeedProfile,
    SpeedProfileSample,
)
from optiline.core.planning.feasibility import (
    FeasibilityIssue,
    FeasibilityReport,
    validate_speed_profile,
)
from optiline.core.planning.lap_time import LapTimingResult, compute_lap_time
from optiline.core.planning.optimisation import (
    RacingLineOptimisationConfig,
    RacingLineResult,
    optimise_racing_line,
)
from optiline.core.planning.path import build_baseline_path
from optiline.core.planning.racing_line import (
    BoundaryOffsetError,
    OffsetBounds,
    RacingLineOptimiserMetadata,
    RacingLinePath,
    baseline_fingerprint,
    build_offset_bounds,
    build_racing_line_path,
    evaluate_periodic_offsets,
)
from optiline.core.planning.racing_line_validation import (
    RacingLineValidationReport,
    validate_racing_line,
)
from optiline.core.planning.speed_profile import build_speed_profile

__all__ = [
    "ILLUSTRATIVE_LATERAL_ACCELERATION_MPS2",
    "PLANNING_SCHEMA_VERSION",
    "BaselinePath",
    "BoundaryOffsetError",
    "FeasibilityIssue",
    "FeasibilityReport",
    "LapTimingResult",
    "OffsetBounds",
    "PathSample",
    "PlanningLimits",
    "RacingLineOptimisationConfig",
    "RacingLineOptimiserMetadata",
    "RacingLinePath",
    "RacingLineResult",
    "RacingLineValidationReport",
    "SpeedProfile",
    "SpeedProfileSample",
    "baseline_fingerprint",
    "build_baseline_path",
    "build_offset_bounds",
    "build_racing_line_path",
    "build_speed_profile",
    "compute_lap_time",
    "evaluate_periodic_offsets",
    "optimise_racing_line",
    "validate_racing_line",
    "validate_speed_profile",
]
