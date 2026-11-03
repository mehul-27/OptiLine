"""M3.1 dynamic single-track bicycle model with linear tyres.

The rear axle is the reference point. Body axes are +x forward and +y left;
heading is counter-clockwise from world +x. The CG is ``l_r`` ahead of the
rear axle and the front axle is ``l_f`` ahead of the CG. Both ``v_x`` and
``v_y`` are rear-axle velocities.

Slip angles use the rear-axle velocity convention:

    beta_f = atan2(v_y + (l_f + l_r) * r, v_x)
    alpha_f = steering - beta_f
    beta_r = atan2(v_y, v_x)
    alpha_r = -beta_r
    F_yf = C_f * alpha_f
    F_yr = C_r * alpha_r

Positive steering, slip angle and lateral force are all to the left. Linear
tyres are valid only for small slip angles and do not model saturation, load
transfer, combined slip or high-slip behaviour.

Derivation of the CG-offset terms:

    v_CG = (v_x, v_y + l_r * r)
    a_CG = (
        v_x_dot - r * v_y - l_r * r^2,
        v_y_dot + r * v_x + l_r * r_dot,
    )

Equating ``m * a_CG`` to the body forces and using
``I_z * r_dot = l_f * F_yf_body - l_r * F_yr`` gives the implemented
``v_x_dot`` and ``v_y_dot`` equations. The front axle velocity is
``(v_x, v_y + (l_f + l_r) * r)`` and the rear axle velocity is
``(v_x, v_y)``.

The longitudinal control is an acceleration-equivalent longitudinal force
command, not the realised body acceleration. The model applies ``m * a_x`` at
the vehicle body and also includes the front tyre's steering-induced
longitudinal force ``-F_yf * sin(steering)``. The realised ``v_x_dot`` therefore
also contains the steering drag, Coriolis and CG-offset terms. At zero
steering, lateral velocity and yaw rate, the command and the realised
longitudinal acceleration coincide.

The low-speed transition blends two states independently advanced by one
timestep from the same initial state, control and timestep; it does not blend
derivatives. The M2.1 kinematic branch uses ``v_y = 0`` and its kinematic yaw
rate, while the dynamic branch uses the six-state model. Heading is blended by
the shortest angular difference.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from optiline.core.simulation.kinematic import normalize_angle_rad, step_kinematic_bicycle
from optiline.core.vehicle.config import VEHICLE_SCHEMA_VERSION, VehicleConfig
from optiline.core.vehicle.dynamic_config import (
    DynamicSimulationConfig,
    DynamicVehicleConfig,
)
from optiline.core.vehicle.dynamic_state import DynamicVehicleState
from optiline.core.vehicle.state import VehicleControl, VehicleState


@dataclass(frozen=True)
class _Derivative:
    x_dot_mps: float
    y_dot_mps: float
    heading_dot_radps: float
    v_x_dot_mps2: float
    v_y_dot_mps2: float
    yaw_acceleration_radps2: float


@dataclass(frozen=True)
class _StateVector:
    x: float
    y: float
    heading: float
    v_x: float
    v_y: float
    yaw_rate: float


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _ensure_finite(values: tuple[float, ...], label: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"dynamic integration produced a non-finite {label}")


def low_speed_blend_factor(
    speed_mps: float,
    config: DynamicVehicleConfig,
) -> float:
    """Return a smooth 0-to-1 blend between kinematic and dynamic behaviour."""

    if not math.isfinite(speed_mps) or speed_mps < 0.0:
        raise ValueError("speed_mps must be finite and non-negative")
    if speed_mps <= config.low_speed_transition_start_mps:
        return 0.0
    if speed_mps >= config.low_speed_transition_end_mps:
        return 1.0
    normalized = (speed_mps - config.low_speed_transition_start_mps) / (
        config.low_speed_transition_end_mps - config.low_speed_transition_start_mps
    )
    return normalized * normalized * (3.0 - 2.0 * normalized)


def slip_angles(
    state: DynamicVehicleState,
    control: VehicleControl,
    config: DynamicVehicleConfig,
) -> tuple[float, float]:
    """Return front and rear slip angles in radians.

    The longitudinal speed used in ``atan2`` is floored at zero so intermediate
    RK4 stages cannot produce a reverse-angle discontinuity. The actual
    low-speed transition is the smooth kinematic/dynamic blend.
    """

    v_x = max(state.v_x_mps, 0.0)
    front_beta = math.atan2(
        state.v_y_mps + config.wheelbase_m * state.yaw_rate_radps,
        v_x,
    )
    rear_beta = math.atan2(state.v_y_mps, v_x)
    return control.steering_angle_rad - front_beta, -rear_beta


def lateral_forces(
    state: DynamicVehicleState,
    control: VehicleControl,
    config: DynamicVehicleConfig,
) -> tuple[float, float]:
    """Return front and rear linear lateral forces in newtons."""

    front_slip, rear_slip = slip_angles(state, control, config)
    return (
        config.front_cornering_stiffness_n_per_rad * front_slip,
        config.rear_cornering_stiffness_n_per_rad * rear_slip,
    )


def _derivative_values(
    x: float,
    y: float,
    heading: float,
    v_x: float,
    v_y: float,
    yaw_rate: float,
    control: VehicleControl,
    config: DynamicVehicleConfig,
) -> _Derivative:
    v_x_safe = max(v_x, 0.0)
    front_beta = math.atan2(
        v_y + config.wheelbase_m * yaw_rate,
        v_x_safe,
    )
    rear_beta = math.atan2(v_y, v_x_safe)
    front_slip = control.steering_angle_rad - front_beta
    rear_slip = -rear_beta

    front_lateral = config.front_cornering_stiffness_n_per_rad * front_slip
    rear_lateral = config.rear_cornering_stiffness_n_per_rad * rear_slip
    cos_steering = math.cos(control.steering_angle_rad)
    sin_steering = math.sin(control.steering_angle_rad)

    front_lateral_body = front_lateral * cos_steering
    front_longitudinal_body = -front_lateral * sin_steering
    longitudinal_force = (
        config.mass_kg * control.longitudinal_acceleration_mps2 + front_longitudinal_body
    )
    lateral_force = front_lateral_body + rear_lateral
    yaw_moment = (
        config.front_axle_distance_m * front_lateral_body
        - config.rear_axle_distance_m * rear_lateral
    )
    yaw_acceleration = yaw_moment / config.yaw_inertia_kg_m2

    # Rear-axle reference: the CG is l_r ahead of the state point, so the
    # rigid-body acceleration includes the CG-offset terms.
    v_x_dot = (
        longitudinal_force / config.mass_kg
        + yaw_rate * v_y
        + yaw_rate * yaw_rate * config.rear_axle_distance_m
    )
    v_y_dot = (
        lateral_force / config.mass_kg
        - yaw_rate * v_x_safe
        - yaw_acceleration * config.rear_axle_distance_m
    )
    x_dot = v_x_safe * math.cos(heading) - v_y * math.sin(heading)
    y_dot = v_x_safe * math.sin(heading) + v_y * math.cos(heading)
    derivative = _Derivative(
        x_dot_mps=x_dot,
        y_dot_mps=y_dot,
        heading_dot_radps=yaw_rate,
        v_x_dot_mps2=v_x_dot,
        v_y_dot_mps2=v_y_dot,
        yaw_acceleration_radps2=yaw_acceleration,
    )
    _ensure_finite(
        (
            derivative.x_dot_mps,
            derivative.y_dot_mps,
            derivative.heading_dot_radps,
            derivative.v_x_dot_mps2,
            derivative.v_y_dot_mps2,
            derivative.yaw_acceleration_radps2,
        ),
        "derivative",
    )
    return derivative


def _validate_step(
    state: DynamicVehicleState,
    control: VehicleControl,
    config: DynamicVehicleConfig,
) -> None:
    if state.v_x_mps > config.max_speed_mps:
        raise ValueError("state v_x_mps exceeds vehicle max_speed_mps")
    if abs(control.steering_angle_rad) > config.max_steering_angle_rad:
        raise ValueError("steering_angle_rad exceeds vehicle steering limit")
    if control.longitudinal_acceleration_mps2 > config.max_acceleration_mps2:
        raise ValueError("longitudinal acceleration exceeds vehicle acceleration limit")
    if control.longitudinal_acceleration_mps2 < -config.max_braking_mps2:
        raise ValueError("longitudinal acceleration exceeds vehicle braking limit")


def _finalize_state(
    x: float,
    y: float,
    heading: float,
    v_x: float,
    v_y: float,
    yaw_rate: float,
    config: DynamicVehicleConfig,
) -> DynamicVehicleState:
    values = (x, y, heading, v_x, v_y, yaw_rate)
    _ensure_finite(values, "state")
    return DynamicVehicleState(
        x_m=x,
        y_m=y,
        heading_rad=normalize_angle_rad(heading),
        v_x_mps=_clamp(v_x, 0.0, config.max_speed_mps),
        v_y_mps=v_y,
        yaw_rate_radps=yaw_rate,
    )


def _kinematic_config(config: DynamicVehicleConfig) -> VehicleConfig:
    return VehicleConfig(
        schema_version=VEHICLE_SCHEMA_VERSION,
        name=config.name,
        model_type="kinematic_bicycle",
        wheelbase_m=config.wheelbase_m,
        max_steering_angle_rad=config.max_steering_angle_rad,
        max_acceleration_mps2=config.max_acceleration_mps2,
        max_braking_mps2=config.max_braking_mps2,
        max_speed_mps=config.max_speed_mps,
    )


def _kinematic_next(
    state: DynamicVehicleState,
    control: VehicleControl,
    config: DynamicVehicleConfig,
    dt_s: float,
) -> tuple[VehicleState, float]:
    kinematic_state = VehicleState(
        x_m=state.x_m,
        y_m=state.y_m,
        heading_rad=state.heading_rad,
        speed_mps=state.v_x_mps,
    )
    next_state = step_kinematic_bicycle(
        kinematic_state,
        control,
        _kinematic_config(config),
        dt_s,
    )
    yaw_rate = (next_state.speed_mps / config.wheelbase_m) * math.tan(control.steering_angle_rad)
    return next_state, yaw_rate


def _kinematic_as_dynamic(
    kinematic: VehicleState,
    yaw_rate: float,
    config: DynamicVehicleConfig,
) -> DynamicVehicleState:
    return _finalize_state(
        kinematic.x_m,
        kinematic.y_m,
        kinematic.heading_rad,
        kinematic.speed_mps,
        0.0,
        yaw_rate,
        config,
    )


def _blend_states(
    kinematic: VehicleState,
    kinematic_yaw_rate: float,
    dynamic: DynamicVehicleState,
    blend: float,
    config: DynamicVehicleConfig,
) -> DynamicVehicleState:
    heading_delta = normalize_angle_rad(dynamic.heading_rad - kinematic.heading_rad)
    heading = normalize_angle_rad(kinematic.heading_rad + blend * heading_delta)
    return _finalize_state(
        x=(1.0 - blend) * kinematic.x_m + blend * dynamic.x_m,
        y=(1.0 - blend) * kinematic.y_m + blend * dynamic.y_m,
        heading=heading,
        v_x=(1.0 - blend) * kinematic.speed_mps + blend * dynamic.v_x_mps,
        v_y=blend * dynamic.v_y_mps,
        yaw_rate=(1.0 - blend) * kinematic_yaw_rate + blend * dynamic.yaw_rate_radps,
        config=config,
    )


def _semi_implicit_step(
    state: DynamicVehicleState,
    control: VehicleControl,
    config: DynamicVehicleConfig,
    dt_s: float,
) -> DynamicVehicleState:
    derivative = _derivative_values(
        state.x_m,
        state.y_m,
        state.heading_rad,
        state.v_x_mps,
        state.v_y_mps,
        state.yaw_rate_radps,
        control,
        config,
    )
    v_x = _clamp(
        state.v_x_mps + dt_s * derivative.v_x_dot_mps2,
        0.0,
        config.max_speed_mps,
    )
    v_y = state.v_y_mps + dt_s * derivative.v_y_dot_mps2
    yaw_rate = state.yaw_rate_radps + dt_s * derivative.yaw_acceleration_radps2
    heading = normalize_angle_rad(state.heading_rad + dt_s * yaw_rate)
    x = state.x_m + dt_s * (v_x * math.cos(heading) - v_y * math.sin(heading))
    y = state.y_m + dt_s * (v_x * math.sin(heading) + v_y * math.cos(heading))
    return _finalize_state(x, y, heading, v_x, v_y, yaw_rate, config)


def _vector_from_state(state: DynamicVehicleState) -> _StateVector:
    return _StateVector(
        x=state.x_m,
        y=state.y_m,
        heading=state.heading_rad,
        v_x=state.v_x_mps,
        v_y=state.v_y_mps,
        yaw_rate=state.yaw_rate_radps,
    )


def _add_scaled(vector: _StateVector, derivative: _Derivative, scale: float) -> _StateVector:
    result = _StateVector(
        x=vector.x + scale * derivative.x_dot_mps,
        y=vector.y + scale * derivative.y_dot_mps,
        heading=vector.heading + scale * derivative.heading_dot_radps,
        v_x=vector.v_x + scale * derivative.v_x_dot_mps2,
        v_y=vector.v_y + scale * derivative.v_y_dot_mps2,
        yaw_rate=vector.yaw_rate + scale * derivative.yaw_acceleration_radps2,
    )
    _ensure_finite(
        (result.x, result.y, result.heading, result.v_x, result.v_y, result.yaw_rate),
        "intermediate state",
    )
    return result


def _rk4_step(
    state: DynamicVehicleState,
    control: VehicleControl,
    config: DynamicVehicleConfig,
    dt_s: float,
) -> DynamicVehicleState:
    initial = _vector_from_state(state)
    k1 = _derivative_values(
        initial.x,
        initial.y,
        initial.heading,
        initial.v_x,
        initial.v_y,
        initial.yaw_rate,
        control,
        config,
    )
    second = _add_scaled(initial, k1, dt_s / 2.0)
    k2 = _derivative_values(
        second.x,
        second.y,
        second.heading,
        second.v_x,
        second.v_y,
        second.yaw_rate,
        control,
        config,
    )
    third = _add_scaled(initial, k2, dt_s / 2.0)
    k3 = _derivative_values(
        third.x,
        third.y,
        third.heading,
        third.v_x,
        third.v_y,
        third.yaw_rate,
        control,
        config,
    )
    fourth = _add_scaled(initial, k3, dt_s)
    k4 = _derivative_values(
        fourth.x,
        fourth.y,
        fourth.heading,
        fourth.v_x,
        fourth.v_y,
        fourth.yaw_rate,
        control,
        config,
    )
    final = _StateVector(
        x=initial.x
        + dt_s / 6.0 * (k1.x_dot_mps + 2.0 * k2.x_dot_mps + 2.0 * k3.x_dot_mps + k4.x_dot_mps),
        y=initial.y
        + dt_s / 6.0 * (k1.y_dot_mps + 2.0 * k2.y_dot_mps + 2.0 * k3.y_dot_mps + k4.y_dot_mps),
        heading=initial.heading
        + dt_s
        / 6.0
        * (
            k1.heading_dot_radps
            + 2.0 * k2.heading_dot_radps
            + 2.0 * k3.heading_dot_radps
            + k4.heading_dot_radps
        ),
        v_x=initial.v_x
        + dt_s
        / 6.0
        * (k1.v_x_dot_mps2 + 2.0 * k2.v_x_dot_mps2 + 2.0 * k3.v_x_dot_mps2 + k4.v_x_dot_mps2),
        v_y=initial.v_y
        + dt_s
        / 6.0
        * (k1.v_y_dot_mps2 + 2.0 * k2.v_y_dot_mps2 + 2.0 * k3.v_y_dot_mps2 + k4.v_y_dot_mps2),
        yaw_rate=initial.yaw_rate
        + dt_s
        / 6.0
        * (
            k1.yaw_acceleration_radps2
            + 2.0 * k2.yaw_acceleration_radps2
            + 2.0 * k3.yaw_acceleration_radps2
            + k4.yaw_acceleration_radps2
        ),
    )
    return _finalize_state(
        final.x,
        final.y,
        final.heading,
        final.v_x,
        final.v_y,
        final.yaw_rate,
        config,
    )


def step_dynamic_bicycle(
    state: DynamicVehicleState,
    control: VehicleControl,
    config: DynamicVehicleConfig,
    simulation_config: DynamicSimulationConfig,
) -> DynamicVehicleState:
    """Advance the dynamic model by one configured fixed step."""

    _validate_step(state, control, config)
    blend = low_speed_blend_factor(state.v_x_mps, config)
    if blend <= 0.0:
        kinematic, kinematic_yaw_rate = _kinematic_next(
            state,
            control,
            config,
            simulation_config.dt_s,
        )
        return _kinematic_as_dynamic(kinematic, kinematic_yaw_rate, config)

    if simulation_config.integration_method == "semi_implicit_euler":
        dynamic = _semi_implicit_step(
            state,
            control,
            config,
            simulation_config.dt_s,
        )
    else:
        dynamic = _rk4_step(state, control, config, simulation_config.dt_s)

    if blend >= 1.0:
        return dynamic

    kinematic, kinematic_yaw_rate = _kinematic_next(
        state,
        control,
        config,
        simulation_config.dt_s,
    )
    return _blend_states(
        kinematic,
        kinematic_yaw_rate,
        dynamic,
        blend,
        config,
    )


def run_dynamic_steps(
    initial_state: DynamicVehicleState,
    controls: Iterable[VehicleControl],
    config: DynamicVehicleConfig,
    simulation_config: DynamicSimulationConfig,
) -> tuple[DynamicVehicleState, ...]:
    """Run a fixed-step dynamic sequence and return every state."""

    states = [initial_state]
    state = initial_state
    for control in controls:
        state = step_dynamic_bicycle(state, control, config, simulation_config)
        states.append(state)
    return tuple(states)
