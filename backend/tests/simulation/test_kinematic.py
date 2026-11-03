from __future__ import annotations

import math

import pytest

from optiline.core.simulation.kinematic import (
    normalize_angle_rad,
    run_fixed_steps,
    step_kinematic_bicycle,
)
from optiline.core.vehicle.config import VEHICLE_SCHEMA_VERSION, VehicleConfig
from optiline.core.vehicle.state import VehicleControl, VehicleState


def make_config(**overrides: object) -> VehicleConfig:
    data: dict[str, object] = {
        "schema_version": VEHICLE_SCHEMA_VERSION,
        "name": "test vehicle",
        "model_type": "kinematic_bicycle",
        "wheelbase_m": 2.5,
        "max_steering_angle_rad": 0.5,
        "max_acceleration_mps2": 5.0,
        "max_braking_mps2": 10.0,
        "max_speed_mps": 50.0,
    }
    data.update(overrides)
    return VehicleConfig.model_validate(data)


def make_state(
    *,
    x_m: float = 0.0,
    y_m: float = 0.0,
    heading_rad: float = 0.0,
    speed_mps: float = 0.0,
) -> VehicleState:
    return VehicleState(
        x_m=x_m,
        y_m=y_m,
        heading_rad=heading_rad,
        speed_mps=speed_mps,
    )


def make_control(
    *,
    steering_angle_rad: float = 0.0,
    longitudinal_acceleration_mps2: float = 0.0,
) -> VehicleControl:
    return VehicleControl(
        steering_angle_rad=steering_angle_rad,
        longitudinal_acceleration_mps2=longitudinal_acceleration_mps2,
    )


def test_straight_line_constant_speed() -> None:
    config = make_config()
    initial = make_state(speed_mps=10.0)
    controls = [make_control() for _ in range(100)]
    final = run_fixed_steps(initial, controls, config, 0.1)[-1]
    assert final.x_m == pytest.approx(100.0)
    assert final.y_m == pytest.approx(0.0)
    assert final.heading_rad == pytest.approx(0.0)
    assert final.speed_mps == pytest.approx(10.0)


def test_straight_line_constant_acceleration_matches_discrete_update() -> None:
    config = make_config()
    initial = make_state()
    controls = [make_control(longitudinal_acceleration_mps2=1.0) for _ in range(10)]
    final = run_fixed_steps(initial, controls, config, 0.1)[-1]
    expected_x = sum((step * 0.1) * 0.1 for step in range(1, 11))
    assert final.speed_mps == pytest.approx(1.0)
    assert final.x_m == pytest.approx(expected_x)
    assert final.y_m == pytest.approx(0.0)
    assert final.heading_rad == pytest.approx(0.0)


@pytest.mark.parametrize("steering", [0.2, -0.2])
def test_positive_and_negative_steering_turn_in_opposite_directions(
    steering: float,
) -> None:
    config = make_config()
    initial = make_state(speed_mps=10.0)
    controls = [make_control(steering_angle_rad=steering) for _ in range(10)]
    final = run_fixed_steps(initial, controls, config, 0.1)[-1]
    assert final.heading_rad * steering > 0.0
    assert final.y_m * steering > 0.0
    assert final.speed_mps == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("heading_rad", "expected_x", "expected_y", "expected_heading"),
    [
        (0.0, 0.1, 0.0, 0.0),
        (math.pi / 2.0, 0.0, 0.1, math.pi / 2.0),
        (math.pi, -0.1, 0.0, -math.pi),
    ],
)
def test_heading_uses_ccw_world_convention(
    heading_rad: float,
    expected_x: float,
    expected_y: float,
    expected_heading: float,
) -> None:
    config = make_config()
    result = step_kinematic_bicycle(
        make_state(heading_rad=heading_rad, speed_mps=1.0),
        make_control(),
        config,
        0.1,
    )
    assert result.x_m == pytest.approx(expected_x)
    assert result.y_m == pytest.approx(expected_y)
    assert result.heading_rad == pytest.approx(expected_heading)


def test_zero_speed_zero_controls_is_stationary() -> None:
    config = make_config()
    initial = make_state()
    result = step_kinematic_bicycle(initial, make_control(), config, 0.1)
    assert result == initial


def test_zero_speed_positive_acceleration_moves_forward() -> None:
    config = make_config()
    initial = make_state()
    result = step_kinematic_bicycle(
        initial,
        make_control(longitudinal_acceleration_mps2=1.0),
        config,
        0.1,
    )
    assert result.speed_mps == pytest.approx(0.1)
    assert result.x_m == pytest.approx(0.01)
    assert result.y_m == pytest.approx(0.0)
    assert result.heading_rad == pytest.approx(0.0)


def test_steering_limits_are_enforced() -> None:
    config = make_config()
    state = make_state(speed_mps=10.0)
    step_kinematic_bicycle(
        state,
        make_control(steering_angle_rad=config.max_steering_angle_rad),
        config,
        0.1,
    )
    step_kinematic_bicycle(
        state,
        make_control(steering_angle_rad=-config.max_steering_angle_rad),
        config,
        0.1,
    )
    with pytest.raises(ValueError, match="steering"):
        step_kinematic_bicycle(
            state,
            make_control(steering_angle_rad=config.max_steering_angle_rad + 1e-9),
            config,
            0.1,
        )
    with pytest.raises(ValueError, match="steering"):
        step_kinematic_bicycle(
            state,
            make_control(steering_angle_rad=-config.max_steering_angle_rad - 1e-9),
            config,
            0.1,
        )


def test_acceleration_and_braking_limits_are_enforced() -> None:
    config = make_config()
    state = make_state(speed_mps=10.0)
    step_kinematic_bicycle(
        state,
        make_control(longitudinal_acceleration_mps2=config.max_acceleration_mps2),
        config,
        0.1,
    )
    step_kinematic_bicycle(
        state,
        make_control(longitudinal_acceleration_mps2=-config.max_braking_mps2),
        config,
        0.1,
    )
    with pytest.raises(ValueError, match="acceleration"):
        step_kinematic_bicycle(
            state,
            make_control(
                longitudinal_acceleration_mps2=config.max_acceleration_mps2 + 1e-9,
            ),
            config,
            0.1,
        )
    with pytest.raises(ValueError, match="braking"):
        step_kinematic_bicycle(
            state,
            make_control(
                longitudinal_acceleration_mps2=-config.max_braking_mps2 - 1e-9,
            ),
            config,
            0.1,
        )


@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, -math.inf],
)
def test_invalid_state_values_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        VehicleState(x_m=value, y_m=0.0, heading_rad=0.0, speed_mps=0.0)
    with pytest.raises(ValueError, match="finite"):
        VehicleState(x_m=0.0, y_m=0.0, heading_rad=value, speed_mps=0.0)
    with pytest.raises(ValueError, match="finite"):
        VehicleState(x_m=0.0, y_m=0.0, heading_rad=0.0, speed_mps=value)


def test_negative_speed_state_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_state(speed_mps=-1.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_invalid_control_values_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        VehicleControl(
            steering_angle_rad=value,
            longitudinal_acceleration_mps2=0.0,
        )
    with pytest.raises(ValueError, match="finite"):
        VehicleControl(
            steering_angle_rad=0.0,
            longitudinal_acceleration_mps2=value,
        )


@pytest.mark.parametrize("dt_s", [0.0, -0.1, math.nan, math.inf])
def test_invalid_timestep_is_rejected(dt_s: float) -> None:
    with pytest.raises(ValueError, match="dt_s"):
        step_kinematic_bicycle(
            make_state(speed_mps=1.0),
            make_control(),
            make_config(),
            dt_s,
        )


def test_state_speed_above_config_limit_is_rejected() -> None:
    config = make_config(max_speed_mps=10.0)
    with pytest.raises(ValueError, match="max_speed"):
        step_kinematic_bicycle(
            make_state(speed_mps=10.1),
            make_control(),
            config,
            0.1,
        )


def test_max_speed_is_clamped() -> None:
    config = make_config(max_speed_mps=50.0)
    first = step_kinematic_bicycle(
        make_state(speed_mps=49.5),
        make_control(longitudinal_acceleration_mps2=5.0),
        config,
        1.0,
    )
    second = step_kinematic_bicycle(
        first,
        make_control(longitudinal_acceleration_mps2=5.0),
        config,
        1.0,
    )
    assert first.speed_mps == pytest.approx(50.0)
    assert second.speed_mps == pytest.approx(50.0)


def test_braking_at_zero_does_not_reverse() -> None:
    config = make_config()
    result = step_kinematic_bicycle(
        make_state(),
        make_control(longitudinal_acceleration_mps2=-config.max_braking_mps2),
        config,
        0.1,
    )
    assert result == make_state()


def test_deterministic_repeated_runs() -> None:
    config = make_config()
    initial = make_state(speed_mps=5.0)
    controls = [
        make_control(
            steering_angle_rad=0.1 if index % 2 == 0 else -0.1,
            longitudinal_acceleration_mps2=1.0 if index % 3 else -2.0,
        )
        for index in range(20)
    ]
    first = run_fixed_steps(initial, controls, config, 0.05)
    second = run_fixed_steps(initial, controls, config, 0.05)
    assert first == second


def test_heading_is_normalized() -> None:
    assert normalize_angle_rad(0.0) == pytest.approx(0.0)
    assert normalize_angle_rad(math.pi) == pytest.approx(-math.pi)
    assert normalize_angle_rad(3.0 * math.pi) == pytest.approx(-math.pi)
    assert normalize_angle_rad(-3.0 * math.pi) == pytest.approx(-math.pi)
    with pytest.raises(ValueError, match="finite"):
        normalize_angle_rad(math.nan)


def test_heading_rate_is_timestep_independent() -> None:
    config = make_config()
    initial = make_state(speed_mps=10.0)
    control = make_control(steering_angle_rad=0.2)
    coarse = run_fixed_steps(initial, [control] * 10, config, 0.1)[-1]
    fine = run_fixed_steps(initial, [control] * 100, config, 0.01)[-1]
    assert coarse.heading_rad == pytest.approx(fine.heading_rad, abs=1e-12)
    # Position uses first-order semi-implicit integration, so the coarse and
    # fine paths differ by a bounded amount that shrinks with the timestep.
    assert coarse.x_m == pytest.approx(fine.x_m, rel=0.1)
    assert coarse.y_m == pytest.approx(fine.y_m, rel=0.1)


def test_constant_steering_circle_reference() -> None:
    wheelbase = 2.5
    radius = 5.0
    config = make_config(wheelbase_m=wheelbase, max_steering_angle_rad=0.5)
    steering = math.atan(wheelbase / radius)
    speed = 10.0
    angular_rate = speed / wheelbase * math.tan(steering)
    dt_s = 0.001
    step_count = round(2.0 * math.pi / (angular_rate * dt_s))
    initial = make_state(speed_mps=speed)
    controls = [make_control(steering_angle_rad=steering)] * step_count
    states = run_fixed_steps(initial, controls, config, dt_s)
    radius_errors = [abs(math.hypot(state.x_m, state.y_m - radius) - radius) for state in states]
    final = states[-1]
    assert max(radius_errors) < 0.01
    assert math.hypot(final.x_m, final.y_m) < 0.02
    assert abs(final.heading_rad) < 0.01
