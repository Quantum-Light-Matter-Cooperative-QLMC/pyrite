# Refactored marimo analysis app

Place `analysis_app.py` beside the existing `_design.py` and `_widgets.py`, and place the
`analysis_ui/` package beside it:

```text
src/pyrite/apps/
├── analysis_app.py
├── compare_app.py
├── pixel_app.py
├── _design.py
├── _widgets.py
└── analysis_ui/
    ├── __init__.py
    ├── axes.py
    ├── controls.py
    ├── data.py
    ├── interactive.py
    ├── models.py
    ├── pixels.py
    └── views/
        ├── __init__.py
        ├── cases.py
        ├── common.py
        ├── detectors.py
        ├── dimension.py
        ├── energy.py
        ├── materials.py
        ├── optimize.py
        └── pixels.py
```

Run it as before:

```bash
marimo run src/pyrite/apps/analysis_app.py
```

## Structure

- `analysis_app.py`: reactive wiring, state callbacks, and final layout only.
- `pixel_app.py`: stored pixel-detector observations (`pyrite app pixels`); its own
  material/face/profile picker resolves a checkpoint stem without loading the checkpoint.
- `compare_app.py`: case basket across checkpoints plus cross-material comparison
  (`pyrite app compare`); its own picker loads the checkpoint the basket draws from.
- `pixels.py`: observation discovery/loading, controls, and image resolution for `pixel_app.py`.
- `data.py`: checkpoint selection/loading and emission selection.
- `controls.py`: composite `mo.ui.dictionary` controls.
- `axes.py`: validated linear/log domains, including automatic positive lower limits.
- `interactive.py`: top-level Altair selection widgets that require the default transformer.
- `views/`: ordinary Python rendering functions grouped by application section.
  `views/pixels.py`, `views/cases.py`, and `views/materials.py` serve the pixel and compare
  apps and are imported by submodule path, not from the `views` package.

These apps ship in the wheel and may be launched from any working directory via
the `pyrite app` commands. For edit/watch workflows, pass the packaged path shown
above directly to marimo.
