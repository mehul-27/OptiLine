# OptiLine — Project Blueprint

**Status:** Initial working specification  
**Purpose:** Source of truth for architecture and incremental implementation. Update this document deliberately as decisions evolve.

## 1. Product goal

OptiLine is a browser-based 2D vehicle-dynamics and racing-line optimisation simulator. A user creates or imports a track, calibrates its physical scale, selects a vehicle, and calculates and visualises a fast, physically feasible lap from a top-down view.

The product combines:
- Track geometry authoring and calibration.
- A deterministic vehicle simulation.
- A mathematical racing-line and speed-profile optimiser.
- A reinforcement-learning (RL) driving agent for experimentation and comparison.
- Top-down visualisation and telemetry.

OptiLine is an engineering simulator with a visual interface, not primarily a racing game.

## 2. Guiding principles

1. Validate deterministic geometry and physics before integrating RL.
2. Keep rendering separate from authoritative physics state.
3. Keep track, vehicle, simulation, optimisation and RL responsibilities separate.
4. Vehicle and simulation parameters must be configuration-driven, not scattered hard-coded constants.
5. Use SI units internally: metres, seconds, kilograms, radians, metres per second and newtons. Convert to km/h and other display units only in the UI.
6. Make simulation runs reproducible by default. Randomness must be explicit and seedable.
7. Never claim real-world accuracy solely because the track is scaled. Accuracy depends on the vehicle model and its validation.
8. Prefer small, testable, reversible changes. Explain new dependencies and avoid broad rewrites.
9. Keep the UI evolvable; refine it alongside working functionality rather than overdesigning it upfront.
10. Avoid accounts, cloud persistence and unnecessary infrastructure in the initial version.

## 3. Initial technology direction

Approved technology direction (M0, refined after the pre-M1 reuse audit):
- Frontend: React + TypeScript + Vite.
- 2D rendering: start with native HTML Canvas 2D. Add Konva/react-konva only if editor interaction complexity justifies it; evaluate PixiJS only if a rendering benchmark shows Canvas is insufficient.
- Backend: Python 3.12 with FastAPI, introduced when the HTTP boundary is required.
- Schemas: Pydantic v2 for versioned track and vehicle contracts.
- Track geometry: NumPy for numeric work and Shapely 2.2 for general planar validity, containment, distance and self-intersection checks. Track-specific corridor, start/finish and calibration rules remain OptiLine code.
- Numerics: NumPy from M1.1; add SciPy only when a concrete spline, spatial or optimisation need is established.
- Track and vehicle simulation: custom, deterministic vehicle model; use a general 2D physics engine only if collision/contact requirements justify it.
- Optimisation: evaluate CasADi for constrained optimal-control formulations at M4. CasADi is LGPL-3.0+ and IPOPT is EPL-2.0; review distribution obligations before adoption.
- RL interface: Gymnasium, introduced after deterministic simulation and validation exist.
- RL algorithms: Stable-Baselines3 as the initial candidate.
- Persistence: JSON track and vehicle configuration files initially; consider SQLite only when the use case requires it.
- Python tooling: uv for environment and lockfile management; pytest, ruff, and mypy for quality gates.
- Frontend tooling: npm, Vitest/React Testing Library, Playwright, ESLint, and Prettier, introduced with the frontend.

M0 installed only pytest, ruff, and mypy. M1.1 adds NumPy, Pydantic and Shapely for the track schema and geometry slice. FastAPI, SciPy, CasADi, Gymnasium, Stable-Baselines3 and the frontend stack remain deferred until their milestones.

Do not install all proposed dependencies at once. Select dependencies when a milestone requires them and explain why. Record adopted, deferred and rejected dependencies in `DEPENDENCIES.md`.

## 4. System boundaries

### Frontend
Responsible for track editing interactions, configuration forms, rendering, playback controls and displaying telemetry. It must not mutate or calculate authoritative physics state.

### Track domain
Stores calibrated geometry and metadata independent of how the track was authored. Image files are optional reference layers, not the authoritative track definition.

### Vehicle simulation
Consumes a track-independent vehicle configuration, state and control inputs. Advances the vehicle using a defined simulation timestep and returns state/telemetry. It must be deterministic for the same initial state, inputs and configuration.

### Optimiser
Consumes a track and vehicle model/constraints, and produces a candidate path, speed profile and predicted lap time. Validate candidate feasibility with the simulation; never present an unvalidated path as guaranteed optimal.

### RL environment
Wraps the same authoritative simulation with observations, actions, rewards, termination conditions and reproducible reset behaviour. It must not fork or silently redefine vehicle physics.

### API and persistence
Expose explicit versioned data contracts between browser and Python backend. Keep persistence local and simple initially.

## 5. Track creation and calibration

Support three authoring modes:
1. Manual drawing and editing.
2. Track construction from straights, arcs/turns and other primitives.
3. Image upload as a reference layer, with manual tracing of inner and outer track boundaries. Automatic boundary extraction is a later enhancement.

The normalised track representation must be independent of the authoring method and include, as applicable:
- Track identifier, name and schema version.
- Inner and outer boundaries, or an equivalent validated drivable corridor.
- Centreline derived from geometry, not treated as the only safety boundary.
- Start/finish location and direction of travel.
- Physical scale/calibration metadata and its source.
- Derived lap length in metres.
- Geometry validation status and warnings.

Calibration requirements:
- Support a user-provided known distance between two points or a known track dimension.
- Store the calibration transform and source.
- Clearly indicate uncalibrated tracks; do not report physically meaningful speed or lap-time claims for them.
- Check for self-intersections, gaps, invalid widths and ambiguous start/finish geometry.
- Keep screen pixels separate from world coordinates. Use metres in simulation.
- Track length must be calculated from the actual chosen path/geometry, not the image dimensions.
- An image alone does not establish physical scale or guarantee accurate boundaries.

M1.1 track contract:
- The persisted representation is an OptiLine-owned JSON document with a schema version, track identity/name, coordinate convention (`x_right_y_up_ccw_rad`), units, outer/inner boundaries, start/finish line and direction, and calibration provenance.
- Boundaries are closed polylines stored without repeating the first point. Canonical orientation is outer counter-clockwise and inner clockwise; the loader normalizes orientation and records a warning if it reverses a boundary.
- The start/finish is a directed segment spanning the corridor plus a unit forward vector.
- A calibrated track is stored in metres and may report physical lengths. An uncalibrated track is editable but uses `units="unspecified"`, is not simulation-ready and must not report physical lap lengths or speeds.
- Derived length, corridor width, curvature and validation results are computed on load and are not trusted from disk.
- TUM CSV, CommonRoad and OpenDRIVE are future import/export adapters only. They are not the internal schema.
- M1.1 supports the known-distance calibration path. Known track dimensions and image tracing remain later work.
- M1.1 supports corridors whose derived nearest-point centerline is continuous. A centerline segment longer than 5x the expected uniform spacing is reported as `CENTERLINE_DISCONTINUITY` and the track is not simulation-ready. General non-convex centerline generation remains later work.

## 6. Vehicle model progression

Start with a kinematic bicycle model for early integration and visual validation. Progress to a dynamic bicycle model with mass, yaw inertia, tyre-force approximation, steering limits, longitudinal acceleration/braking and lateral dynamics. Consider a four-wheel model only when tests and product goals justify its complexity.

Vehicle configuration should cover relevant parameters such as:
- Mass and yaw inertia.
- Wheelbase and axle geometry.
- Steering limits and steering response.
- Engine or motor power/torque characteristics and drivetrain limits.
- Braking limits.
- Tyre/grip parameters and a documented tyre-force model.
- Aerodynamic drag and downforce where applicable.
- Model type and configuration schema version.

Do not imply that a vehicle preset is empirically accurate unless its parameters have a traceable source and validation. Label estimated/default presets as illustrative.

## 7. Optimisation approach

Use a staged approach:
1. Produce a valid geometric baseline path.
2. Calculate a feasible speed profile subject to curvature, grip, acceleration and braking limits.
3. Optimise path and speed together where the selected model and solver support it.
4. Validate track-boundary and vehicle constraints in the authoritative simulation.
5. Report lap time, feasibility, solver status and relevant assumptions.

Evaluate CasADi and an appropriate constrained optimal-control formulation before committing. A geometric shortest path is not necessarily a minimum-time racing line. The objective is minimum lap time under explicit vehicle and track constraints, not minimum distance alone.

Provide a benchmark that is deterministic and repeatable. Distinguish solver convergence from proof of a global optimum.

M1.1 does not implement the optimiser. The first optimiser work remains a later, bounded experiment after the track and vehicle interfaces are stable. Use existing racing-line projects as references first; verify licence, provenance and compatibility before copying or porting code. Record the solver version, options, tolerances, warm-start state and validation results with every benchmark. Do not copy LGPL TUM code into the core without legal review.

## 8. Reinforcement learning

RL is complementary to the optimiser, not a replacement for it. The optimiser supplies a benchmark and may optionally provide a reference trajectory.

Initial RL controls may include steering, throttle and brake. Observations may include speed, heading error, lateral offset, local curvature and relevant vehicle state. Define rewards and termination rules carefully to discourage track departures, collisions, unstable behaviour and simulator exploitation.

The RL environment must use the same simulation model as validation and optimisation wherever practical. Start RL only after deterministic vehicle behaviour, track boundaries and episode termination are tested. Record random seeds, configuration versions, reward definitions and evaluation results.

## 9. Units, timing and telemetry

- Internal SI units only.
- Use a fixed, documented simulation timestep initially; make it configurable only if necessary.
- Distinguish simulated time from wall-clock compute time.
- Lap time is elapsed simulated seconds between valid start/finish crossings.
- Report speed in km/h in the UI by converting from m/s.
- Record distance along the lap, speed, steering, throttle, brake, lateral acceleration and other state supported by the model.
- Flag invalid laps and explain why (for example, track departure or failure to cross the finish).
- Do not infer actual real-world performance from visual plausibility alone.

## 10. Incremental milestones and acceptance criteria

### M0 — Repository and architecture
- Establish this blueprint as the working specification.
- Inspect current files and clean obsolete starter files without deleting Git history.
- Document the chosen stack and development commands.
- Acceptance: clean project skeleton, documented decisions, no unnecessary dependencies.

### M1 — Track domain and editor foundation
- **M1.1 (approved current slice):** define the versioned track schema, coordinate system, known-distance calibration, core geometry validation and analytic fixtures. Do not include vehicle dynamics, the optimiser, RL, image recognition or the full UI.
- **M1.1 acceptance:** create, save and reload analytic circle/oval tracks; verify calibration and known distances; validate boundaries, corridor width and start/finish; compute length within documented tolerances; round-trip JSON.
- **M1.2 (later):** implement the minimal manual editor and its save/reload workflow.
- **M1 acceptance:** tracks from the editor use the same normalised geometry contract; verify known distances and calculated length.

### M2 — Deterministic vehicle prototype
- Implement a kinematic bicycle model and a fixed-step simulation loop.
- Add configuration-driven vehicle parameters and unit tests.
- Acceptance: repeatable runs, predictable straight-line movement and steering response, no dependence on rendering frame rate.

### M3 — Dynamic model and validation
- Add dynamic bicycle behaviour incrementally, including documented tyre/grip assumptions.
- Acceptance: tests for braking, turning, grip limits and numerical stability; all model limitations documented.

### M4 — Baseline path and speed optimiser
- Produce a track-valid path and a physically feasible speed profile.
- Add lap timing and independent feasibility checks.
- Evaluate CasADi/IPOPT in a bounded experiment only after the vehicle and track interfaces are stable; record solver metadata and validate every candidate lap in the authoritative simulation.
- Acceptance: valid laps on simple test circuits; optimised time is compared with a baseline and no improvement is claimed without measurement.

### M5 — Telemetry and comparison
- Show the vehicle, line, speed, lap time and basic telemetry in the browser.
- Compare baseline and optimised runs.
- Acceptance: displayed values agree with backend run data and can be reproduced.

### M6 — RL integration
- Wrap the tested simulation as a Gymnasium environment and train an initial agent.
- Evaluate against a fixed benchmark across repeatable trials.
- Acceptance: documented learning/evaluation metrics; no claim of superiority without evidence.

### M7 — Image tracing and quality improvements
- Support image references and manual boundary tracing robustly; evaluate assisted extraction later.
- Improve editing, calibration feedback, saving, errors and usability.
- Acceptance: tracks from different authoring modes produce the same normalised geometry contract.

Milestones can be reordered only with a stated reason. Do not implement the entire roadmap in one pass.

## 11. Testing and quality gates

- Unit-test geometry, coordinate transforms, unit conversions and lap timing.
- Test schema-version rejection, unknown-field rejection and JSON round-trip behavior.
- Test calibration with known distances, forward/inverse transforms and missing/invalid calibration.
- Test closed boundaries, self-intersections, inner/outer ordering, corridor width and start/finish faults.
- Test derived-centerline continuity and ensure unsupported non-convex corridors are not marked simulation-ready.
- Test analytic circle/oval lengths, widths and curvature within documented numerical tolerances; derive tolerances from discretization rather than tightening them arbitrarily.
- Test deterministic physics with fixed initial states and inputs.
- Test track validation for invalid geometry and boundary departures.
- Test optimiser feasibility independently of its objective score.
- Test API data contracts and schema validation.
- Use integration tests before RL training.
- Include a simple known track fixture with explicit dimensions.
- Record known limitations and avoid fabricating benchmark results.
- Review dependency licences, provenance and attribution before copying or porting third-party code.

## 12. Collaboration and coding-agent rules

When implementing:
1. Read this blueprint and relevant existing files first.
2. State the current milestone and the smallest proposed change.
3. Do not implement future milestones without approval.
4. Do not add a dependency without explaining the need and alternatives.
5. Do not create unnecessary abstractions, folders or modules.
6. Keep physics independent from rendering and RL.
7. Run relevant tests and report actual results; never claim tests passed if they were not run.
8. If a design decision materially conflicts with this blueprint, explain the trade-off and ask before proceeding.
9. At the end of each milestone, summarise files changed, behaviour implemented, tests run, known limitations and the next proposed step.
10. Treat this document as a living specification; propose edits when implementation reveals a meaningful need, rather than silently changing core assumptions.
11. Use existing implementations as references first. Before copying or porting code, verify its licence, provenance, compatibility and attribution requirements, and record the decision in `DEPENDENCIES.md`.
12. Do not reuse unlicensed code. LGPL/EPL/MPL code requires a specific integration decision and legal review before it enters the core.
13. Do not introduce an external simulator or physics engine as the authoritative core without new evidence that its benefits outweigh integration and physics-consistency costs.
14. Keep milestone changes separate and reviewable; do not mix M0 cleanup, M1.1 track work and later experiments in one change set.
15. Do not claim optimality or real-world physical accuracy beyond what tests establish.

## 13. First implementation task

Do not begin by building the full UI, RL agent or optimiser.

First inspect the repository and propose a minimal M0 project skeleton and dependency plan. Preserve Git history. Identify which existing files should be retained, revised or removed. Wait for approval before destructive cleanup or broad implementation.
