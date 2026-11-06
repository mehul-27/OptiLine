# OptiLine Dependency and Licence Register

**Status:** M4.2. Keep this register current when adding, deferring or rejecting a dependency.
**Project licence:** Not selected. Resolve before public distribution or substantial third-party code reuse.

## Adopted

| Dependency | Version | Licence | Purpose | Milestone |
|---|---|---|---|---|
| Python | 3.12.14 | PSF-2.0 | Runtime | M0 |
| uv | 0.12.13 | MIT OR Apache-2.0 | Environment and lockfile management | M0 |
| pytest | 9.2.0 | MIT | Tests | M0 |
| Ruff | 0.17.3 | MIT | Linting and formatting | M0 |
| mypy | 2.4.1 | MIT | Static type checking | M0 |
| NumPy | 2.5.4 | BSD-3-Clause | Numeric geometry, resampling, curvature and width calculations | M1.1 |
| Pydantic | 2.14.2 | MIT | Versioned track and vehicle schemas and validation | M1.1/M2.1 |
| Shapely | 2.2.0 | BSD-3-Clause; bundled GEOS is LGPL-2.1 | Planar validity, containment, self-intersection and distance checks | M1.1 |
| SciPy | 1.18.1 | BSD-3-Clause; bundled OpenBLAS (BSD-3-Clause), LAPACK (BSD-3-Clause-Open-MPI), GCC runtime (GPL-3.0-or-later WITH GCC-exception-3.1) | Periodic spline and constrained SLSQP racing-line optimization | M4.2 |
| types-shapely | 2.1.0.20261030 | Apache-2.0 | Typing stubs for Shapely under strict mypy | M1.1 (dev only) |

M2.1 and M3.1 add no new dependency. The kinematic and dynamic bicycle models use the Python standard library and the existing Pydantic configuration layer. M4.1 added no dependency; baseline planning uses the existing NumPy, Pydantic and Shapely stack. M4.2 adds SciPy 1.18.1 for periodic spline interpolation and SLSQP constrained geometric optimization.

## Deferred

| Dependency | Intended milestone | Licence | Purpose | Reason for deferral |
|---|---|---|---|---|
| FastAPI + uvicorn | M1.2/M5 | MIT / BSD-3-Clause | HTTP API | No API boundary in M1.1 |
| React + TypeScript + Vite | M1.2/M5 | MIT / Apache-2.0 / MIT | Frontend editor and telemetry UI | No frontend in M1.1 |
| react-konva / Konva | Frontend milestone | MIT | Canvas scene graph and editing interactions | Use native Canvas 2D first; add only if interaction complexity justifies it |
| PixiJS | Frontend milestone | MIT | WebGL/WebGPU rendering | Only if a Canvas benchmark shows it is needed |
| CasADi | M4 | LGPL-3.0+ | Optimal-control formulation | Vehicle and track interfaces are not stable yet; legal review required |
| IPOPT | M4 | EPL-2.0 | Nonlinear solver via CasADi | Same as CasADi; review distribution obligations |
| Gymnasium | M6 | MIT | RL environment API | Deterministic simulation and validation must exist first |
| Stable-Baselines3 | M6 | MIT | Initial RL algorithms | Pulls PyTorch and is not needed before M6 |

## Reference-only or rejected

| Candidate | Licence | Status | Reason |
|---|---|---|---|
| TUMFTM/global_racetrajectory_optimization | LGPL-3.0 | Reference/isolated prototype only | Old Python 3.7-era stack; centreline+width model; LGPL obligations |
| TUMFTM/trajectory_planning_helpers | LGPL-3.0 | Reference only | Old stack; LGPL; native `quadprog` dependency |
| CommonRoad Raceline Planner | BSD-3-Clause | Borrow/port only | No minimum-time planner; Python <3.12; provenance/relicensing caveat |
| CommonRoad vehicle models | BSD | Reference/prototype only | Stale 2022 release; research parameters; no fixed-step contract |
| pacejka | MIT | Prototype later at M3 | Simplified tyre model; new/alpha package |
| F1TENTH Gym | MIT | Rejected as foundation | Stale Gym/NumPy pins; different scale/physics |
| racecar_gym | MIT | Rejected as foundation | Stale NumPy/Gymnasium/PyBullet pins; PyBullet physics differs |
| MLTP_AA_ATD_Python | No licence file | Rejected | No reuse permission |
| OpenDRIVE / CommonRoad-IO / scenariogeneration | Free standard / BSD / MPL-2.0 | Future import/export only | Road-network formats, not the internal race-track schema |
| Project Chrono | BSD-3-Clause | Rejected for core | Heavy C++/Python stack; overkill for bicycle models |

## Rules

1. Before adding or copying code, record the licence, version/commit, purpose, compatibility and attribution.
2. Do not copy unlicensed code.
3. LGPL/EPL/MPL code requires a specific integration decision and legal review before entering the core.
4. Do not add a dependency solely because it is popular; state the concrete need and alternative.
5. Update `backend/uv.lock` and this register in the same change.
