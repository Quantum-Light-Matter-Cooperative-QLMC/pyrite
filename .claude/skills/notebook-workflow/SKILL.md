---
name: notebook-workflow
description: Use when changing PyRITE scan, analysis, trace, or validation marimo apps, or cleaning the legacy validation notebook under checks/.
---

# Notebook Workflow

- `scan_app.py`: sweep runner.
- `analysis_app.py`: checkpoint analysis.
- `trace_app.py`: direct transport/lattice viewer.
- `validation_app.py`: validation studies.
- `checks/cxr_analysis_feranchuk.ipynb`: only legacy Jupyter workflow.

Put reusable logic in `src/pyrite/`; keep apps thin. Add no new `.ipynb`
workflows. Keep legacy notebook output-free.

After marimo edits run `uv run marimo check <app.py>`. For legacy notebook run
`pyrite-dev nbqa` before edits and `nbstrip` before handoff. Use
`pyrite-dev verify` for full repository verification.
