---
name: scientific-library
description: "Use when deterministic PyRITE library/API code needs work on vectorization, constants, typing, exports, or scientific docstrings; use monte-carlo or physics-review for those domain concerns."
---

# Scientific Library

## Conventions

- All simulation kernels live in `src/pyrite/`.
- Prefer NumPy/CuPy vectorization over Python loops.
- Avoid OOP unless stateful behavior is required.
- Public APIs require docstrings and type hints.
- Never duplicate physics constants; prefer `scipy.constants` or the project's constants module.

Own library/API quality. Defer stochastic behavior to `monte-carlo`, physics
claims to `physics-review`, benchmarks to `performance`, and CLI surfaces to
`cli-ui-ux`.
