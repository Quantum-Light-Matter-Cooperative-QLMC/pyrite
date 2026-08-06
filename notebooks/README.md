# Refactored marimo analysis app

Place `analysis_app.py` beside the existing `_design.py` and `_widgets.py`, and place the
`analysis_ui/` package beside it:

```text
notebooks/
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
marimo run notebooks/analysis_app.py
```

## Structure

- `analysis_app.py`: reactive wiring, state callbacks, and final layout only.
- `data.py`: checkpoint selection/loading and emission selection.
- `controls.py`: composite `mo.ui.dictionary` controls.
- `axes.py`: validated linear/log domains, including automatic positive lower limits.
- `interactive.py`: top-level Altair selection widgets that require the default transformer.
- `views/`: ordinary Python rendering functions grouped by application section.

The original uploaded file is included as `analysis_app_original.py` for comparison.
