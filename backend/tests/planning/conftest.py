from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from optiline.core.planning import (  # noqa: E402
    PlanningLimits,
    RacingLineOptimisationConfig,
    build_baseline_path,
)
from optiline.core.track.schema import (  # noqa: E402
    Boundary,
    StartFinish,
    TrackBoundaries,
)
from optiline.core.vehicle.dynamic_config import (  # noqa: E402
    DYNAMIC_VEHICLE_SCHEMA_VERSION,
    DynamicVehicleConfig,
)
from tests.fixtures.track_builders import make_circle_track, make_oval_track  # noqa: E402


@pytest.fixture
def vehicle_config() -> DynamicVehicleConfig:
    return DynamicVehicleConfig(
        schema_version=DYNAMIC_VEHICLE_SCHEMA_VERSION,
        name="planning test vehicle",
        model_type="dynamic_single_track",
        mass_kg=1500.0,
        yaw_inertia_kg_m2=2500.0,
        front_axle_distance_m=1.2,
        rear_axle_distance_m=1.6,
        front_cornering_stiffness_n_per_rad=80000.0,
        rear_cornering_stiffness_n_per_rad=80000.0,
        max_steering_angle_rad=0.5,
        max_acceleration_mps2=3.0,
        max_braking_mps2=5.0,
        max_speed_mps=50.0,
        low_speed_transition_start_mps=2.0,
        low_speed_transition_end_mps=4.0,
    )


@pytest.fixture
def planning_limits(vehicle_config: DynamicVehicleConfig) -> PlanningLimits:
    return PlanningLimits.from_dynamic_vehicle_config(vehicle_config)


@pytest.fixture
def circle_track():
    return make_circle_track(count=360)


@pytest.fixture
def oval_track():
    return make_oval_track(count=720)


@pytest.fixture
def circle_path(circle_track):
    return build_baseline_path(circle_track, sample_count=360)


@pytest.fixture
def oval_path(oval_track):
    return build_baseline_path(oval_track, sample_count=720)


@pytest.fixture
def optimisation_config() -> RacingLineOptimisationConfig:
    return RacingLineOptimisationConfig(
        control_point_count=16,
        boundary_margin_m=0.5,
        max_iterations=50,
        ftol=1e-7,
    )


@pytest.fixture
def asymmetric_track():
    import math

    import numpy as np

    base = make_circle_track(count=360)
    angles = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    outer = np.column_stack((60.0 * np.cos(angles), 60.0 * np.sin(angles)))
    inner = np.column_stack((10.0 + 40.0 * np.cos(angles), 40.0 * np.sin(angles)))
    return base.model_copy(
        update={
            "track_id": "asymmetric-circle",
            "name": "Asymmetric circle",
            "boundaries": TrackBoundaries(
                outer=Boundary(points=[(float(x), float(y)) for x, y in outer]),
                inner=Boundary(points=[(float(x), float(y)) for x, y in inner]),
            ),
            "start_finish": StartFinish(
                line=((50.0, 0.0), (60.0, 0.0)),
                forward=(0.0, 1.0),
            ),
        },
    )


@pytest.fixture
def asymmetric_path(asymmetric_track):
    return build_baseline_path(asymmetric_track, sample_count=360)


@pytest.fixture
def narrow_track():
    return make_circle_track(centerline_radius=50.0, half_width=1.0, count=360)


@pytest.fixture
def narrow_path(narrow_track):
    return build_baseline_path(narrow_track, sample_count=360)
