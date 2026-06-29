---
name: scientific-library
description: Apply cxr_mc scientific-library coding conventions. Use when writing or reviewing code in src/cxr_mc/, especially simulation kernels, public APIs, vectorization, constants, and type/docstring consistency.
---

# Scientific Library

## Conventions

- All simulation kernels live in `src/cxr_mc/`.
- Prefer NumPy/CuPy vectorization over Python loops.
- Avoid OOP unless stateful behavior is required.
- Public APIs require docstrings and type hints.
- Never duplicate physics constants; prefer `scipy.constants` or the project's constants module.
