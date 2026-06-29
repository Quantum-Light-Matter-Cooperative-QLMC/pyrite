---
name: notebook-workflow
description: Edit, validate, and sanitize repository notebooks. Use when changing scan.ipynb, analysis.ipynb, or notebooks under checks/; keep notebooks output-free and source logic in src/cxr_mc/.
---

# Notebook Workflow

## Rules

- Do not put reusable physics logic in a notebook cell if it belongs in `src/cxr_mc/`.
- Keep committed notebooks output-stripped.
- Prefer `scripts/dev.py nbqa` before editing notebook code, and `scripts/dev.py nbstrip` after.
- If a notebook starts growing a real workflow, move the reusable parts into source code and leave the notebook as a thin driver.

## Canonical Commands

- `uv run python scripts/dev.py nbqa`
- `uv run python scripts/dev.py nbstrip`
- `uv run python scripts/dev.py verify`
