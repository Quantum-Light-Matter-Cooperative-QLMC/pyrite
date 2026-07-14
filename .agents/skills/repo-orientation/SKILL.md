---
name: repo-orientation
description: Use when locating code, choosing an owning module, assessing cxr-mc repository structure, updating the repository map, or planning a surgical change.
---

# Repo Orientation

## Ground Rules

- Prefer `src/cxr_mc/` for implementation changes.
- Use `tests/` for fast CPU checks.
- Use `checks/` for heavier physics validation anchors.
- Treat `notebooks/scan_app.py`, `notebooks/analysis_app.py`, and
  `notebooks/validation_app.py` as the three marimo application entry points.
- Treat `checks/cxr_analysis_feranchuk.ipynb` as the remaining legacy validation
  notebook, not an application owner.
- Do not rewrite `README.md`, `TODO.md`, or `docs/` unless the task is about those files.

## Canonical Commands


- `uv run python scripts/dev.py repo-map`
- `uv run python scripts/dev.py lint`
- `uv run python scripts/dev.py format`
- `uv run python scripts/dev.py test`
- `uv run python scripts/dev.py verify`

## Optional semantic navigation (Serena)

When Serena is available, prefer its `find_symbol`,
`find_referencing_symbols`, and `get_symbols_overview` tools for code
navigation; use `rename_symbol` for cross-file renames.

Troubleshooting: `serena project health-check`

## Before Changing Code

1. Find the smallest file that owns the behavior.
2. Check whether a helper already exists in `src/cxr_mc/`.
3. Prefer a focused test over a broad refactor.
4. Keep notebook edits limited to presentation or analysis flow.

Regenerate the inventory with `uv run python scripts/dev.py repo-map` after
entry points, packages, or agent-tooling top-levels change.
