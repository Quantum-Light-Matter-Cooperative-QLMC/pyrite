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
    ├── navigation.py
    ├── pickers.py
    ├── pixels.py
    └── views/
        ├── __init__.py
        ├── cases.py
        ├── common.py
        ├── detectors.py
        ├── map.py
        ├── materials.py
        ├── pixels.py
        ├── sidebar.py
        └── spectra.py
```

Run it as before:

```bash
marimo run src/pyrite/apps/analysis_app.py
```

## Structure

- `analysis_app.py`: reactive wiring, state callbacks, and final layout only. Three
  views — **Spectra**, **Map**, **Detectors** — sit behind a dependency-free tab bar
  (`navigation.py`); each view's cells stop unless it is selected. The pickers and the
  shared slice (beam energy, polar tilt, azimuth, thickness) live in `mo.sidebar`, one
  dropdown per name, so a view reruns only when a dimension it reads changes.
- `pixel_app.py`: stored pixel-detector observations (`pyrite app pixels`); its
  Material + Checkpoint picker resolves a checkpoint stem without loading the checkpoint.
- `compare_app.py`: case basket across checkpoints plus cross-material comparison
  (`pyrite app compare`); its own picker loads the checkpoint the basket draws from.
- `pixels.py`: observation discovery/loading, controls, and image resolution for `pixel_app.py`.
- `pickers.py`: the shared Material + Checkpoint picker. The Checkpoint dropdown lists
  every stem with its face (`high_energy (a7b2ce) · flat`) and defaults only to the
  standard flat run; otherwise it opens on a placeholder and nothing loads.
- `data.py`: checkpoint loading, emission selection, and `slice_results` (sidebar slice
  pinning with the watchdog thickness fallback).
- `controls.py`: the sidebar slice, view-local choices (vary radio, component toggles,
  map quantity, detector radio), and one axis-control helper per view
  (`make_spectra_axes`, `make_detector_axes`).
- `axes.py`: validated linear/log domains, including automatic positive lower limits.
- `interactive.py`: the Map view's click-selectable heatmap, which requires the default
  transformer.
- `views/`: ordinary Python rendering functions grouped by application section.
  `views/pixels.py`, `views/cases.py`, and `views/materials.py` serve the pixel and compare
  apps and are imported by submodule path, not from the `views` package.

These apps ship in the wheel and may be launched from any working directory via
the `pyrite app` commands. For edit/watch workflows, pass the packaged path shown
above directly to marimo.
