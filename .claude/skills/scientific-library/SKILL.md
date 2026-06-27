# Coding conventions for cxr_mc's scientific library

# Use this skill when

Use this skill when writing or reviewing any code in `src/cxr_mc/`.

# Conventions

- All simulation kernels live in `src/cxr_mc/`.
- Prefer NumPy/CuPy vectorization over Python loops.
- Avoid OOP unless stateful behavior is required.
- Public APIs require docstrings and type hints.
- Never duplicate physics constants — prefer `scipy.constants` or the project's constants module.
