from __future__ import annotations

import math

import pytest

from optiline.core.simulation.dynamic import run_dynamic_steps, step_dynamic_bicycle
from optiline.core.vehicle.dynamic_config import (
    DYNAMIC_VEHICLE_SCHEMA_VERSION,
    DynamicSimulationConfig,
    DynamicVehicleConfig,
)
from optiline.core.vehicle.dynamic_state import DynamicVehicleState
from optiline.core.vehicle.state import VehicleControl


def make_config() -> DynamicVehicleConfig:
    return DynamicVehicleConfig(
        schema_version=DYNAMIC_VEHICLE_SCHEMA_VERSION,
        name="integration test vehicle",
        model_type="dynamic_single_track",
        mass_kg=1500.0,
        yaw_inertia_kg_m2=2500.0,
        front_axle_distance_m=1.2,
        rear_axle_distance_m=1.6,
        front_cornering_stiffness_n_per_rad=80000.0,
        rear_cornering_stiffness_n_per_rad=80000.0,
        max_steering_angle_rad=0.5,
        max_acceleration_mps2=5.0,
        max_braking_mps2=10.0,
        max_speed_mps=50.0,
        low_speed_transition_start_mps=2.0,
        low_speed_transition_end_mps=4.0,
    )


def make_state() -> DynamicVehicleState:
    return DynamicVehicleState(
        x_m=0.0,
        y_m=0.0,
        heading_rad=0.0,
        v_x_mps=15.0,
        v_y_mps=0.0,
        yaw_rate_radps=0.0,
    )


def make_sim(method: str, dt_s: float) -> DynamicSimulationConfig:
    return DynamicSimulationConfig(dt_s=dt_s, integration_method=method)  # type: ignore[arg-type]


def run(
    initial: DynamicVehicleState,
    control: VehicleControl,
    duration_s: float,
    dt_s: float,
    method: str,
) -> DynamicVehicleState:
    steps = max(1, round(duration_s / dt_s))
    simulation = make_sim(method, duration_s / steps)
    return run_dynamic_steps(
        initial,
        [control] * steps,
        make_config(),
        simulation,
    )[-1]


def position_error(left: DynamicVehicleState, right: DynamicVehicleState) -> float:
    return math.hypot(left.x_m - right.x_m, left.y_m - right.y_m)


def test_rk4_is_more_accurate_than_semi_implicit_for_constant_cornering() -> None:
    initial = make_state()
    control = VehicleControl(steering_angle_rad=0.05, longitudinal_acceleration_mps2=0.0)
    reference = run(initial, control, 2.0, 0.0005, "rk4")
    coarse_rk4 = run(initial, control, 2.0, 0.02, "rk4")
    coarse_euler = run(initial, control, 2.0, 0.02, "semi_implicit_euler")
    rk4_error = position_error(coarse_rk4, reference)
    euler_error = position_error(coarse_euler, reference)
    assert rk4_error < 1e-4
    assert rk4_error < euler_error


@pytest.mark.parametrize("method", ["semi_implicit_euler", "rk4"])
def test_methods_converge_under_timestep_reduction(method: str) -> None:
    initial = make_state()
    control = VehicleControl(steering_angle_rad=0.05, longitudinal_acceleration_mps2=0.0)
    reference = run(initial, control, 2.0, 0.0005, "rk4")
    coarse = run(initial, control, 2.0, 0.04, method)
    fine = run(initial, control, 2.0, 0.01, method)
    assert position_error(fine, reference) < position_error(coarse, reference)


def test_braking_while_cornering_is_finite_for_both_methods() -> None:
    initial = make_state()
    control = VehicleControl(steering_angle_rad=0.05, longitudinal_acceleration_mps2=-3.0)
    euler = run(initial, control, 1.0, 0.01, "semi_implicit_euler")
    rk4 = run(initial, control, 1.0, 0.01, "rk4")
    assert euler.v_x_mps > 0.0
    assert rk4.v_x_mps > 0.0
    assert position_error(euler, rk4) < 0.2
    assert euler.yaw_rate_radps == pytest.approx(rk4.yaw_rate_radps, rel=0.05)


def test_transient_steering_is_finite_for_both_methods() -> None:
    initial = make_state()
    controls = [
        VehicleControl(
            steering_angle_rad=0.1 if index < 100 else -0.1,
            longitudinal_acceleration_mps2=1.0 if index < 50 else -1.0,
        )
        for index in range(200)
    ]
    config = make_config()
    results = {}
    for method in ("semi_implicit_euler", "rk4"):
        simulation = make_sim(method, 0.01)
        states = run_dynamic_steps(initial, controls, config, simulation)
        final = states[-1]
        assert all(
            math.isfinite(value)
            for value in (
                final.x_m,
                final.y_m,
                final.heading_rad,
                final.v_x_mps,
                final.v_y_mps,
                final.yaw_rate_radps,
            )
        )
        results[method] = final
    assert position_error(results["semi_implicit_euler"], results["rk4"]) < 0.5


def test_low_speed_transition_is_finite_for_both_methods() -> None:
    initial = DynamicVehicleState(
        x_m=0.0,
        y_m=0.0,
        heading_rad=0.0,
        v_x_mps=2.5,
        v_y_mps=0.0,
        yaw_rate_radps=0.0,
    )
    control = VehicleControl(steering_angle_rad=0.1, longitudinal_acceleration_mps2=3.0)
    config = make_config()
    for method in ("semi_implicit_euler", "rk4"):
        final = run_dynamic_steps(
            initial,
            [control] * 300,
            config,
            make_sim(method, 0.01),
        )[-1]
        assert final.v_x_mps > 4.0
        assert math.isfinite(final.v_y_mps)
        assert math.isfinite(final.yaw_rate_radps)


@pytest.mark.parametrize("method", ["semi_implicit_euler", "rk4"])
def test_long_run_remains_bounded(method: str) -> None:
    initial = make_state()
    control = VehicleControl(steering_angle_rad=0.05, longitudinal_acceleration_mps2=0.0)
    final = run_dynamic_steps(
        initial,
        [control] * 5000,
        make_config(),
        make_sim(method, 0.01),
    )[-1]
    assert final.v_x_mps > 0.0
    assert abs(final.v_y_mps) < 2.0
    assert abs(final.yaw_rate_radps) < 2.0


@pytest.mark.parametrize("method", ["semi_implicit_euler", "rk4"])
def test_repeated_runs_are_deterministic(method: str) -> None:
    initial = make_state()
    controls = [
        VehicleControl(
            steering_angle_rad=0.08 if index % 2 == 0 else -0.08,
            longitudinal_acceleration_mps2=1.0 if index % 3 else -2.0,
        )
        for index in range(100)
    ]
    config = make_config()
    simulation = make_sim(method, 0.01)
    first = run_dynamic_steps(initial, controls, config, simulation)
    second = run_dynamic_steps(initial, controls, config, simulation)
    assert first == second


def test_step_respects_state_speed_limit() -> None:
    config = make_config()
    state = DynamicVehicleState(
        x_m=0.0,
        y_m=0.0,
        heading_rad=0.0,
        v_x_mps=config.max_speed_mps + 1.0,
        v_y_mps=0.0,
        yaw_rate_radps=0.0,
    )
    with pytest.raises(ValueError, match="max_speed"):
        step_dynamic_bicycle(
            state,
            VehicleControl(steering_angle_rad=0.0, longitudinal_acceleration_mps2=0.0),
            config,
            make_sim("rk4", 0.01),
        )
