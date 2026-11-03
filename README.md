# OptiLine

OptiLine is a browser-based 2D vehicle-dynamics and racing-line optimisation simulator. It combines track authoring and calibration, a deterministic vehicle simulation, a minimum-lap-time optimiser, and an experimental reinforcement-learning agent.

The project is an engineering simulator with a visual interface, not a racing game. M1.1 track schema and geometry, the M2.1 kinematic bicycle and the M3.1 dynamic single-track prototype are implemented. There is no optimiser, RL environment or UI yet.

## Documentation

- [PROJECT_BLUEPRINT.md](PROJECT_BLUEPRINT.md) — authoritative specification, milestones, and acceptance criteria.
- [RULES.md](RULES.md) — architectural and change-management rules.
- [DEPENDENCIES.md](DEPENDENCIES.md) — adopted, deferred and rejected dependencies with licences.

## Prerequisites

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/)
- Node.js 22 and npm for the future frontend

## Backend setup

```bash
cd backend
uv sync
```

Run the quality gates:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Apply formatting with `uv run ruff format .`.

## Planned commands

These become available as the relevant milestones are implemented:

```bash
# API
uv run uvicorn optiline.api.app:app --reload --port 8000

# Frontend
cd frontend
npm ci
npm run dev
```

## Repository layout

```text
backend/            Python package and tests
backend/src/optiline/core/track/
                    Track schema, geometry, calibration, validation and I/O
backend/src/optiline/core/vehicle/
                    Kinematic/dynamic configuration, state and control types
backend/src/optiline/core/simulation/
                    Fixed-step kinematic and dynamic single-track simulations
frontend/           React + TypeScript editor (M1.2+)
DEPENDENCIES.md     Dependency and licence register
PROJECT_BLUEPRINT.md
RULES.md
```

## Development rules

Read `PROJECT_BLUEPRINT.md` before changing architecture. Implement only the current milestone, explain new dependencies, keep physics independent of rendering and RL, and report the tests actually run. Internal calculations use SI units; display conversions belong in the UI.

## Status

M1.1 covers the versioned track schema, known-distance calibration, core geometry validation and analytic fixtures. It supports corridors whose derived nearest-point centerline is continuous; detected centerline discontinuities are not simulation-ready. M2.1 adds a deterministic, configuration-driven rear-axle kinematic bicycle. M3.1 adds a six-state rear-axle dynamic single-track model with linear cornering stiffness, a smooth low-speed kinematic/dynamic blend, and both RK4 and semi-implicit Euler integration; RK4 is selected for accuracy. The dynamic state uses rear-axle velocities; its longitudinal control is an acceleration-equivalent force command, not a realised acceleration. Nonlinear tyres, load transfer, aerodynamics, the optimiser, RL environment and UI are not implemented. No licence has been selected yet.
