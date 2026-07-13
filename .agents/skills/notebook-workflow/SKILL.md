---
name: notebook-workflow
description: Use when changing the scan, analysis, or validation marimo apps, or cleaning the remaining legacy validation notebook in checks/.
---

# Notebook Workflow

## Owners

- `notebooks/scan_app.py`: interactive sweep runner.
- `notebooks/analysis_app.py`: checkpoint analysis and visualization.
- `notebooks/validation_app.py`: validation-study interface.
- `checks/cxr_analysis_feranchuk.ipynb`: remaining legacy validation notebook;
  it is the only notebook subject to nbQA and output stripping.

## Rules

- Put reusable science and data logic in `src/cxr_mc/`; keep apps thin.
- Keep the legacy notebook output-free and avoid adding new `.ipynb` workflows.
- Validate marimo apps with `uv run marimo check <app.py>` when they change.
- Run `uv run python scripts/dev.py nbqa` before editing the legacy notebook and
  `uv run python scripts/dev.py nbstrip` before handoff.
- Use `uv run python scripts/dev.py verify` for repository verification.
