"""Deterministic fixed-step kinematic bicycle simulation."""

from __future__ import annotations

import math
from collections.abc import Iterable

from optiline.core.vehicle.config import VehicleConfig
from optiline.core.vehicle.state import VehicleControl, VehicleState


def normalize_angle_rad(angle_rad: float) -> float:
    """Normalize an angle to the half-open interval [-pi, pi)."""

    if not math.isfinite(angle_rad):
        raise ValueError("angle_rad must be finite")
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _validate_timestep(dt_s: float) -> None:
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")


def step_kinematic_bicycle(
    state: VehicleState,
    control: VehicleControl,
    config: VehicleConfig,
    dt_s: float,
) -> VehicleState:
    """Advance a rear-axle kinematic bicycle by one fixed timestep.

    The semi-implicit update is::

        v_next = clamp(v + a * dt, 0, v_max)
        heading_next = wrap(heading + v_next / wheelbase * tan(steering) * dt)
        x_next = x + v_next * cos(heading_next) * dt
        y_next = y + v_next * sin(heading_next) * dt

    This is an explicit, first-order integration. Steering is an instantaneous
    angle, not a steering-rate state. Reverse motion, tyre forces and actuator
    dynamics are outside this model.
    """

    _validate_timestep(dt_s)

    if state.speed_mps > config.max_speed_mps:
        raise ValueError("state speed exceeds vehicle max_speed_mps")
    if abs(control.steering_angle_rad) > config.max_steering_angle_rad:
        raise ValueError("steering_angle_rad exceeds vehicle steering limit")
    if control.longitudinal_acceleration_mps2 > config.max_acceleration_mps2:
        raise ValueError("longitudinal acceleration exceeds vehicle acceleration limit")
    if control.longitudinal_acceleration_mps2 < -config.max_braking_mps2:
        raise ValueError("longitudinal acceleration exceeds vehicle braking limit")

    speed_next = min(
        max(
            state.speed_mps + control.longitudinal_acceleration_mps2 * dt_s,
            0.0,
        ),
        config.max_speed_mps,
    )
    heading_next = normalize_angle_rad(
        state.heading_rad
        + (speed_next / config.wheelbase_m) * math.tan(control.steering_angle_rad) * dt_s,
    )
    x_next = state.x_m + speed_next * math.cos(heading_next) * dt_s
    y_next = state.y_m + speed_next * math.sin(heading_next) * dt_s

    return VehicleState(
        x_m=x_next,
        y_m=y_next,
        heading_rad=heading_next,
        speed_mps=speed_next,
    )


def run_fixed_steps(
    initial_state: VehicleState,
    controls: Iterable[VehicleControl],
    config: VehicleConfig,
    dt_s: float,
) -> tuple[VehicleState, ...]:
    """Run a fixed-step sequence and return the initial state plus each result."""

    _validate_timestep(dt_s)
    states = [initial_state]
    state = initial_state
    for control in controls:
        state = step_kinematic_bicycle(state, control, config, dt_s)
        states.append(state)
    return tuple(states)
