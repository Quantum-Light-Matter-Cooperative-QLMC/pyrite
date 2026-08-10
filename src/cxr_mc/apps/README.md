# Refactored marimo analysis app

Place `analysis_app.py` beside the existing `_design.py` and `_widgets.py`, and place the
`analysis_ui/` package beside it:

```text
src/cxr_mc/apps/
├── analysis_app.py
├── _design.py
├── _widgets.py
└── analysis_ui/
    ├── __init__.py
    ├── axes.py
    ├── controls.py
    ├── data.py
    ├── interactive.py
    ├── models.py
    └── views/
        ├── __init__.py
        ├── cases.py
        ├── common.py
        ├── detectors.py
        ├── dimension.py
        ├── energy.py
        ├── materials.py
        └── optimize.py
```

Run it as before:

```bash
marimo run src/cxr_mc/apps/analysis_app.py
```

## Structure

- `analysis_app.py`: reactive wiring, state callbacks, and final layout only.
- `data.py`: checkpoint selection/loading and emission selection.
- `controls.py`: composite `mo.ui.dictionary` controls.
- `axes.py`: validated linear/log domains, including automatic positive lower limits.
- `interactive.py`: top-level Altair selection widgets that require the default transformer.
- `views/`: ordinary Python rendering functions grouped by application section.

These apps ship in the wheel and may be launched from any working directory via
the `pyrite app` commands. For edit/watch workflows, pass the packaged path shown
above directly to marimo.
