---
name: repo-orientation
description: Find the right file, command, or module in cxr_mc before editing. Use when locating code, choosing an owning module, assessing repository structure, or making a surgical change.
---

# Repo Orientation

## Ground Rules

- Prefer `src/cxr_mc/` for implementation changes.
- Use `tests/` for fast CPU checks.
- Use `checks/` for heavier physics validation anchors.
- Treat `scan.ipynb` as the sweep runner and `analysis.ipynb` as the viz notebook.
- Do not rewrite `README.md`, `TODO.md`, or `docs/` unless the task is about those files.

## Canonical Commands

- `uv run python scripts/dev.py repo-map`
- `uv run python scripts/dev.py lint`
- `uv run python scripts/dev.py format`
- `uv run python scripts/dev.py test`
- `uv run python scripts/dev.py verify`

## Before Changing Code

1. Find the smallest file that owns the behavior.
2. Check whether a helper already exists in `src/cxr_mc/`.
3. Prefer a focused test over a broad refactor.
4. Keep notebook edits limited to presentation or analysis flow.
