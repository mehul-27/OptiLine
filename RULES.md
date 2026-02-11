# OptiLine Core Rules

1. No module inside /core may depend on /rl.
2. Rendering must never modify physics state.
3. All vehicle parameters must come from config files.
4. No hardcoded physics constants inside logic files.
5. Deterministic systems must be validated before RL integration.
6. One module = one responsibility.
7. No architecture refactor without discussion.
8. Keep simulation deterministic unless explicitly randomized.
