---
name: notebook-workflow
description: Use when changing cxr-mc scan, analysis, trace, or validation marimo apps, or cleaning the legacy validation notebook under checks/.
---

# Notebook Workflow

- `scan_app.py`: sweep runner.
- `analysis_app.py`: checkpoint analysis.
- `trace_app.py`: direct transport/lattice viewer.
- `validation_app.py`: validation studies.
- `checks/cxr_analysis_feranchuk.ipynb`: only legacy Jupyter workflow.

Put reusable logic in `src/cxr_mc/`; keep apps thin. Add no new `.ipynb`
workflows. Keep legacy notebook output-free.

After marimo edits run `uv run marimo check <app.py>`. For legacy notebook run
`scripts/dev.py nbqa` before edits and `nbstrip` before handoff. Use
`scripts/dev.py verify` for full repository verification.
