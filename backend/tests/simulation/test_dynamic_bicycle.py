from __future__ import annotations

import math

import pytest

from optiline.core.simulation.dynamic import (
    lateral_forces,
    low_speed_blend_factor,
    run_dynamic_steps,
    slip_angles,
    step_dynamic_bicycle,
)
from optiline.core.vehicle.dynamic_config import (
    DYNAMIC_VEHICLE_SCHEMA_VERSION,
    DynamicSimulationConfig,
    DynamicVehicleConfig,
)
from optiline.core.vehicle.dynamic_state import DynamicVehicleState
from optiline.core.vehicle.state import VehicleControl


def make_config(**overrides: object) -> DynamicVehicleConfig:
    data: dict[str, object] = {
        "schema_version": DYNAMIC_VEHICLE_SCHEMA_VERSION,
        "name": "dynamic test vehicle",
        "model_type": "dynamic_single_track",
        "mass_kg": 1500.0,
        "yaw_inertia_kg_m2": 2500.0,
        "front_axle_distance_m": 1.2,
        "rear_axle_distance_m": 1.6,
        "front_cornering_stiffness_n_per_rad": 80000.0,
        "rear_cornering_stiffness_n_per_rad": 80000.0,
        "max_steering_angle_rad": 0.5,
        "max_acceleration_mps2": 5.0,
        "max_braking_mps2": 10.0,
        "max_speed_mps": 50.0,
        "low_speed_transition_start_mps": 2.0,
        "low_speed_transition_end_mps": 4.0,
    }
    data.update(overrides)
    return DynamicVehicleConfig.model_validate(data)


def make_sim(method: str = "rk4", dt_s: float = 0.01) -> DynamicSimulationConfig:
    return DynamicSimulationConfig(dt_s=dt_s, integration_method=method)  # type: ignore[arg-type]


def make_state(
    *,
    x_m: float = 0.0,
    y_m: float = 0.0,
    heading_rad: float = 0.0,
    v_x_mps: float = 15.0,
    v_y_mps: float = 0.0,
    yaw_rate_radps: float = 0.0,
) -> DynamicVehicleState:
    return DynamicVehicleState(
        x_m=x_m,
        y_m=y_m,
        heading_rad=heading_rad,
        v_x_mps=v_x_mps,
        v_y_mps=v_y_mps,
        yaw_rate_radps=yaw_rate_radps,
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


def run(
    initial: DynamicVehicleState,
    control: VehicleControl,
    config: DynamicVehicleConfig,
    *,
    steps: int = 200,
    dt_s: float = 0.01,
    method: str = "rk4",
) -> DynamicVehicleState:
    simulation = make_sim(method, dt_s)
    return run_dynamic_steps(
        initial,
        [control] * steps,
        config,
        simulation,
    )[-1]


@pytest.mark.parametrize("method", ["semi_implicit_euler", "rk4"])
def test_straight_line_zero_steering(method: str) -> None:
    config = make_config()
    initial = make_state(v_x_mps=10.0)
    final = run(initial, make_control(), config, steps=100, method=method)
    assert final.x_m == pytest.approx(10.0)
    assert final.y_m == pytest.approx(0.0, abs=1e-9)
    assert final.heading_rad == pytest.approx(0.0, abs=1e-9)
    assert final.v_x_mps == pytest.approx(10.0)
    assert final.v_y_mps == pytest.approx(0.0, abs=1e-9)
    assert final.yaw_rate_radps == pytest.approx(0.0, abs=1e-9)


def test_acceleration_and_braking_change_longitudinal_speed() -> None:
    config = make_config()
    accelerated = run(
        make_state(v_x_mps=5.0),
        make_control(longitudinal_acceleration_mps2=2.0),
        config,
        steps=100,
    )
    braked = run(
        accelerated,
        make_control(longitudinal_acceleration_mps2=-3.0),
        config,
        steps=100,
    )
    assert accelerated.v_x_mps > 5.0
    assert braked.v_x_mps < accelerated.v_x_mps
    assert braked.v_x_mps >= 0.0


def test_longitudinal_control_is_acceleration_equivalent_force() -> None:
    config = make_config()
    dt_s = 0.01
    initial = make_state(v_x_mps=10.0)
    straight = step_dynamic_bicycle(
        initial,
        make_control(longitudinal_acceleration_mps2=2.0),
        config,
        make_sim("rk4", dt_s),
    )
    assert straight.v_x_mps == pytest.approx(10.0 + 2.0 * dt_s)
    assert straight.v_y_mps == pytest.approx(0.0)
    assert straight.yaw_rate_radps == pytest.approx(0.0)

    turning = step_dynamic_bicycle(
        initial,
        make_control(steering_angle_rad=0.1),
        config,
        make_sim("rk4", dt_s),
    )
    assert turning.v_x_mps < initial.v_x_mps


@pytest.mark.parametrize("steering", [0.05, -0.05])
def test_left_and_right_cornering(steering: float) -> None:
    config = make_config()
    initial = make_state()
    final = run(initial, make_control(steering_angle_rad=steering), config)
    assert final.yaw_rate_radps * steering > 0.0
    assert final.heading_rad * steering > 0.0
    assert final.v_y_mps * steering < 0.0


def test_slip_angle_and_lateral_force_signs() -> None:
    config = make_config()
    initial = make_state()
    left = make_control(steering_angle_rad=0.05)
    front_slip, rear_slip = slip_angles(initial, left, config)
    front_force, rear_force = lateral_forces(initial, left, config)
    assert front_slip > 0.0
    assert rear_slip == pytest.approx(0.0)
    assert front_force > 0.0
    assert rear_force == pytest.approx(0.0)

    settled = run(initial, left, config)
    front_slip, rear_slip = slip_angles(settled, left, config)
    front_force, rear_force = lateral_forces(settled, left, config)
    assert front_slip > 0.0
    assert rear_slip > 0.0
    assert front_force > 0.0
    assert rear_force > 0.0

    right = make_control(steering_angle_rad=-0.05)
    right_settled = run(initial, right, config)
    front_slip, rear_slip = slip_angles(right_settled, right, config)
    assert front_slip < 0.0
    assert rear_slip < 0.0


def test_understeer_and_oversteer_stiffness_response() -> None:
    baseline = make_config()
    front_soft = make_config(front_cornering_stiffness_n_per_rad=40000.0)
    rear_soft = make_config(rear_cornering_stiffness_n_per_rad=40000.0)
    front_stiff = make_config(front_cornering_stiffness_n_per_rad=160000.0)
    rear_stiff = make_config(rear_cornering_stiffness_n_per_rad=160000.0)
    control = make_control(steering_angle_rad=0.05)
    initial = make_state()

    baseline_yaw = run(initial, control, baseline).yaw_rate_radps
    front_soft_yaw = run(initial, control, front_soft).yaw_rate_radps
    rear_soft_yaw = run(initial, control, rear_soft).yaw_rate_radps
    front_stiff_yaw = run(initial, control, front_stiff).yaw_rate_radps
    rear_stiff_yaw = run(initial, control, rear_stiff).yaw_rate_radps

    assert front_soft_yaw < baseline_yaw
    assert rear_soft_yaw > baseline_yaw
    assert front_stiff_yaw > baseline_yaw
    assert rear_stiff_yaw < baseline_yaw


def test_zero_speed_zero_controls_is_stationary() -> None:
    config = make_config()
    initial = make_state(v_x_mps=0.0)
    final = step_dynamic_bicycle(initial, make_control(), config, make_sim())
    assert final == initial


@pytest.mark.parametrize("method", ["semi_implicit_euler", "rk4"])
def test_from_rest_accelerates_and_stays_finite(method: str) -> None:
    config = make_config()
    initial = make_state(v_x_mps=0.0)
    final = run(
        initial,
        make_control(
            steering_angle_rad=0.05,
            longitudinal_acceleration_mps2=2.0,
        ),
        config,
        steps=300,
        method=method,
    )
    assert final.v_x_mps > 4.0
    assert math.isfinite(final.x_m)
    assert math.isfinite(final.y_m)
    assert math.isfinite(final.v_y_mps)
    assert math.isfinite(final.yaw_rate_radps)


def test_braking_to_rest_does_not_reverse() -> None:
    config = make_config()
    final = run(
        make_state(v_x_mps=8.0),
        make_control(
            steering_angle_rad=0.05,
            longitudinal_acceleration_mps2=-3.0,
        ),
        config,
        steps=400,
    )
    assert final.v_x_mps == pytest.approx(0.0)
    assert final.v_y_mps == pytest.approx(0.0, abs=1e-9)
    assert final.yaw_rate_radps == pytest.approx(0.0, abs=1e-9)


def test_hard_braking_across_lower_transition_does_not_reverse() -> None:
    config = make_config()
    states = run_dynamic_steps(
        make_state(v_x_mps=2.5),
        [
            make_control(
                steering_angle_rad=0.1,
                longitudinal_acceleration_mps2=-config.max_braking_mps2,
            )
        ]
        * 100,
        config,
        make_sim("rk4", 0.01),
    )
    assert all(state.v_x_mps >= 0.0 for state in states)
    assert all(
        math.isfinite(value)
        for state in states
        for value in (
            state.x_m,
            state.y_m,
            state.heading_rad,
            state.v_x_mps,
            state.v_y_mps,
            state.yaw_rate_radps,
        )
    )
    assert states[-1].v_x_mps == pytest.approx(0.0)
    assert states[-1].v_y_mps == pytest.approx(0.0, abs=1e-9)
    assert states[-1].yaw_rate_radps == pytest.approx(0.0, abs=1e-9)


def test_steering_and_acceleration_limits_are_enforced() -> None:
    config = make_config()
    state = make_state()
    simulation = make_sim()
    step_dynamic_bicycle(
        state,
        make_control(steering_angle_rad=config.max_steering_angle_rad),
        config,
        simulation,
    )
    step_dynamic_bicycle(
        state,
        make_control(
            steering_angle_rad=-config.max_steering_angle_rad,
            longitudinal_acceleration_mps2=-config.max_braking_mps2,
        ),
        config,
        simulation,
    )
    with pytest.raises(ValueError, match="steering"):
        step_dynamic_bicycle(
            state,
            make_control(steering_angle_rad=config.max_steering_angle_rad + 1e-9),
            config,
            simulation,
        )
    with pytest.raises(ValueError, match="acceleration"):
        step_dynamic_bicycle(
            state,
            make_control(
                longitudinal_acceleration_mps2=config.max_acceleration_mps2 + 1e-9,
            ),
            config,
            simulation,
        )
    with pytest.raises(ValueError, match="braking"):
        step_dynamic_bicycle(
            state,
            make_control(
                longitudinal_acceleration_mps2=-config.max_braking_mps2 - 1e-9,
            ),
            config,
            simulation,
        )


def test_state_and_control_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_state(v_x_mps=-1.0)
    with pytest.raises(ValueError, match="finite"):
        make_state(v_y_mps=math.nan)
    with pytest.raises(ValueError, match="finite"):
        VehicleControl(steering_angle_rad=math.inf, longitudinal_acceleration_mps2=0.0)
    with pytest.raises(ValueError, match="finite"):
        VehicleControl(steering_angle_rad=0.0, longitudinal_acceleration_mps2=math.nan)


@pytest.mark.parametrize("method", ["semi_implicit_euler", "rk4"])
def test_nonfinite_integration_result_is_rejected(method: str) -> None:
    config = make_config()
    with pytest.raises(ValueError, match="non-finite"):
        step_dynamic_bicycle(
            make_state(v_x_mps=15.0, v_y_mps=1e308, yaw_rate_radps=1e308),
            make_control(steering_angle_rad=0.1),
            config,
            make_sim(method, 0.01),
        )


def test_state_speed_above_dynamic_limit_is_rejected() -> None:
    config = make_config(max_speed_mps=10.0)
    with pytest.raises(ValueError, match="max_speed"):
        step_dynamic_bicycle(
            make_state(v_x_mps=10.1),
            make_control(),
            config,
            make_sim(),
        )


def test_heading_wrapping_keeps_state_bounded() -> None:
    config = make_config()
    final = run(
        make_state(),
        make_control(steering_angle_rad=0.2),
        config,
        steps=1000,
    )
    assert -math.pi <= final.heading_rad < math.pi
    assert math.isfinite(final.v_y_mps)
    assert math.isfinite(final.yaw_rate_radps)


def test_low_speed_blend_factor_shape() -> None:
    config = make_config()
    assert low_speed_blend_factor(0.0, config) == pytest.approx(0.0)
    assert low_speed_blend_factor(2.0, config) == pytest.approx(0.0)
    assert low_speed_blend_factor(3.0, config) == pytest.approx(0.5)
    assert low_speed_blend_factor(4.0, config) == pytest.approx(1.0)
    assert low_speed_blend_factor(5.0, config) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-negative"):
        low_speed_blend_factor(-1.0, config)


@pytest.mark.parametrize("speed", [1.9, 2.0, 2.1, 3.0, 3.9, 4.0, 4.1])
def test_low_speed_step_is_finite_and_below_threshold_is_kinematic(
    speed: float,
) -> None:
    config = make_config()
    initial = make_state(v_x_mps=speed)
    final = step_dynamic_bicycle(
        initial,
        make_control(steering_angle_rad=0.1),
        config,
        make_sim(),
    )
    assert math.isfinite(final.x_m)
    assert math.isfinite(final.y_m)
    assert math.isfinite(final.v_y_mps)
    assert math.isfinite(final.yaw_rate_radps)
    if speed <= config.low_speed_transition_start_mps:
        assert final.v_y_mps == pytest.approx(0.0)
        expected_yaw = (final.v_x_mps / config.wheelbase_m) * math.tan(0.1)
        assert final.yaw_rate_radps == pytest.approx(expected_yaw)


def test_low_speed_transition_is_continuous_at_start() -> None:
    config = make_config()
    control = make_control(steering_angle_rad=0.1)
    below = step_dynamic_bicycle(
        make_state(v_x_mps=config.low_speed_transition_start_mps - 1e-6),
        control,
        config,
        make_sim(),
    )
    above = step_dynamic_bicycle(
        make_state(v_x_mps=config.low_speed_transition_start_mps + 1e-6),
        control,
        config,
        make_sim(),
    )
    assert below.x_m == pytest.approx(above.x_m, abs=1e-3)
    assert below.y_m == pytest.approx(above.y_m, abs=1e-3)
    assert below.heading_rad == pytest.approx(above.heading_rad, abs=1e-3)
    assert below.v_x_mps == pytest.approx(above.v_x_mps, abs=1e-3)
    assert below.v_y_mps == pytest.approx(above.v_y_mps, abs=1e-3)
    assert below.yaw_rate_radps == pytest.approx(above.yaw_rate_radps, abs=1e-3)


def test_low_speed_transition_is_continuous_at_end() -> None:
    config = make_config()
    control = make_control(steering_angle_rad=0.1)
    below = step_dynamic_bicycle(
        make_state(v_x_mps=config.low_speed_transition_end_mps - 1e-6),
        control,
        config,
        make_sim(),
    )
    above = step_dynamic_bicycle(
        make_state(v_x_mps=config.low_speed_transition_end_mps + 1e-6),
        control,
        config,
        make_sim(),
    )
    assert below.x_m == pytest.approx(above.x_m, abs=1e-3)
    assert below.y_m == pytest.approx(above.y_m, abs=1e-3)
    assert below.heading_rad == pytest.approx(above.heading_rad, abs=1e-3)
    assert below.v_x_mps == pytest.approx(above.v_x_mps, abs=1e-3)
    assert below.v_y_mps == pytest.approx(above.v_y_mps, abs=1e-3)
    assert below.yaw_rate_radps == pytest.approx(above.yaw_rate_radps, abs=1e-3)


def test_repeated_low_speed_transitions_are_finite() -> None:
    config = make_config()
    controls = [
        make_control(
            steering_angle_rad=0.05,
            longitudinal_acceleration_mps2=5.0 if (index // 100) % 2 == 0 else -5.0,
        )
        for index in range(400)
    ]
    final = run_dynamic_steps(
        make_state(v_x_mps=1.5),
        controls,
        config,
        make_sim("rk4", 0.01),
    )[-1]
    assert 0.0 <= final.v_x_mps <= config.max_speed_mps
    assert math.isfinite(final.v_y_mps)
    assert math.isfinite(final.yaw_rate_radps)


def test_extreme_but_valid_controls_stay_finite() -> None:
    config = make_config()
    final = step_dynamic_bicycle(
        make_state(v_x_mps=10.0),
        make_control(
            steering_angle_rad=config.max_steering_angle_rad,
            longitudinal_acceleration_mps2=config.max_acceleration_mps2,
        ),
        config,
        make_sim(),
    )
    assert final.v_x_mps <= config.max_speed_mps
    assert math.isfinite(final.v_y_mps)
    assert math.isfinite(final.yaw_rate_radps)


def test_braking_while_cornering_remains_finite() -> None:
    config = make_config()
    final = run(
        make_state(),
        make_control(
            steering_angle_rad=0.05,
            longitudinal_acceleration_mps2=-3.0,
        ),
        config,
        steps=100,
    )
    assert final.v_x_mps < 15.0
    assert final.yaw_rate_radps > 0.0
    assert math.isfinite(final.v_y_mps)
