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

- `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py repo-map`
- `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py lint`
- `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py format`
- `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test`
- `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py verify`

## Navigation

1. Read `docs/repo_map.md` for ownership and layer boundaries.
2. Use Tokensave for indexed search, callers/callees, impact, dependencies,
   branch context, and affected-test selection.
3. Use Serena for symbol-precise references, renames, and LSP diagnostics when
   its language server is healthy.
4. Query `.tokensave/tokensave.db` when a structural query lacks a native tool.
5. Use direct source reads or `rg` for exact text, non-code, generated files,
   runtime artifacts, or unindexed details.

Use Context7 only for current external-library documentation. Headroom handles
context/model transport; RTK filters shell output. Neither is a code index.

Troubleshooting: `rtk serena project health-check`

## Before Changing Code

1. Find the smallest file that owns the behavior.
2. Check whether a helper already exists in `src/cxr_mc/`.
3. Prefer a focused test over a broad refactor.
4. Keep notebook edits limited to presentation or analysis flow.

Regenerate inventory with the canonical `repo-map` command after entry points,
packages, or agent-tooling top-levels change.
