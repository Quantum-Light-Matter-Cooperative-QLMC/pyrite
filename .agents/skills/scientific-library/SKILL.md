---
name: scientific-library
description: Use when writing or reviewing cxr-mc library code, especially simulation kernels, public APIs, vectorization, constants, typing, and scientific docstrings.
---

# Scientific Library

## Conventions

- All simulation kernels live in `src/cxr_mc/`.
- Prefer NumPy/CuPy vectorization over Python loops.
- Avoid OOP unless stateful behavior is required.
- Public APIs require docstrings and type hints.
- Never duplicate physics constants; prefer `scipy.constants` or the project's constants module.
