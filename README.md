# OptiLine

OptiLine is a browser-based 2D vehicle-dynamics and racing-line optimisation simulator. It combines track authoring and calibration, a deterministic vehicle simulation, a minimum-lap-time optimiser, and an experimental reinforcement-learning agent.

The project is an engineering simulator with a visual interface, not a racing game. M0 repository and Python project setup is implemented. Track, vehicle, optimiser, RL and UI work are not implemented yet.

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
frontend/           React + TypeScript editor (M1.2+)
DEPENDENCIES.md     Dependency and licence register
PROJECT_BLUEPRINT.md
RULES.md
```

## Development rules

Read `PROJECT_BLUEPRINT.md` before changing architecture. Implement only the current milestone, explain new dependencies, keep physics independent of rendering and RL, and report the tests actually run. Internal calculations use SI units; display conversions belong in the UI.

## Status

M0 establishes the repository and Python project skeleton. No track or vehicle implementation exists yet. No licence has been selected yet.
