---
name: notebook-workflow
description: Use when changing PyRITE scan, analysis, pixel, compare, trace, or validation marimo apps, or cleaning the legacy validation notebook under checks/.
---

# Notebook Workflow

This skill owns packaged Marimo applications and the legacy Jupyter check,
**not** public scientific notebooks in `docs/notebooks/`. Use
`tutorial-authoring` for those, with a separate canonical source/export policy.

- `scan_app.py`: sweep runner.
- `analysis_app.py`: checkpoint analysis.
- `pixel_app.py`: stored pixel-detector observations.
- `compare_app.py`: case basket and cross-material comparison.
- `trace_app.py`: direct transport/lattice viewer.
- `validation_app.py`: validation studies.
- `channeling_app.py`: quantum channeling solver demo.
- `checks/cxr_analysis_feranchuk.ipynb`: only legacy Jupyter workflow.

Put reusable logic in `src/pyrite/`; keep apps thin. Add no new `.ipynb`
workflows. Keep legacy notebook output-free.

After marimo edits run `uv run marimo check <app.py>`. For legacy notebook run
`pyrite-dev nbqa` before edits and `nbstrip` before handoff. Use
the affected app's test module; add reusable-logic consumers only when changed.
Follow `AGENTS.md` "Choose tests by impact"; `apps` is for shared app changes.
`pyrite-dev verify` checks static repository gates only, with no tests.
