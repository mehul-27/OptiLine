# OptiLine Core Rules

## Architecture

1. The `optiline.core` package must not depend on `optiline.rl`, `optiline.api`, or frontend code.
2. Rendering must never modify authoritative physics state.
3. The simulation package is the only authoritative physics implementation. The API, UI, optimiser, and RL environment must call it rather than duplicate or silently redefine vehicle physics.
4. All vehicle parameters must come from configuration files.
5. No hardcoded physics constants inside logic files.
6. Deterministic systems must be validated before RL integration.
7. One module = one responsibility.
8. Keep the simulation deterministic unless randomness is explicit and seedable.
9. Track and vehicle schemas are versioned and validated; do not accept unknown schema versions or silently drop fields.

## Change management

10. No architecture refactor without discussion.
11. Do not create folders, abstractions, or modules unless the current milestone requires them.
12. Do not modify multiple modules in one step.
13. Always explain why a new dependency is required and what alternatives were considered.
14. Never assume architecture beyond the current structure.
15. Ask before introducing abstractions.
16. Avoid rewriting working code unless necessary.
17. Keep changes incremental and reversible.
18. Do not introduce an external simulator or physics engine as the authoritative core without new evidence that its benefits outweigh integration and physics-consistency costs.
19. Do not reuse third-party code without a compatible licence and recorded provenance; LGPL/EPL/MPL code requires a specific integration decision and legal review.
20. Do not claim optimality or real-world physical accuracy beyond what tests establish.
21. Keep milestone changes separate and reviewable; do not mix M0 cleanup, M1.1 track work and later experiments in one change set.
