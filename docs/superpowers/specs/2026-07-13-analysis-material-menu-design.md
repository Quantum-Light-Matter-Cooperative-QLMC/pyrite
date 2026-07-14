# Analysis material menu design

## Goal

Make the analysis app's material control reflect checkpoints currently present
in `checkpoints/`, while keeping every configured material visible. A configured
material without a current checkpoint must be greyed out and unselectable.

## Data and selection

- Discover only top-level `checkpoints/*.pkl` files; archived and reproduction
  caches are not analysis-material checkpoints.
- A pickle stem is available only when it is also in `cxr_mc.config.MATERIALS`.
  Unknown files and quick-checkpoint stems do not create menu entries.
- Display configured materials in their registry order, using `MATERIAL_LABELS`
  where available. Available entries are selectable; all other configured
  materials remain in the menu as disabled entries.
- Keep the existing CLI, environment, and persisted-default precedence for the
  requested initial material. If that material has no checkpoint, select the
  first available configured material. If none are available, render the control
  disabled with no selected material.

## UI

Marimo's dropdown API can disable an entire control but not individual options.
The app will therefore use a small native HTML `select` component that sends
the selected material back through marimo's normal UI-element protocol. This
preserves reactive loading while allowing disabled entries to be shown in the
menu. The app will load a checkpoint only for a selected available material.

## Testing and validation

Put checkpoint discovery and initial-selection resolution in a pure helper in
`cxr_mc.analyze`, with unit tests covering configured and unknown pickle stems,
disabled configured entries, requested available materials, and unavailable
default fallback. Keep the notebook test limited to static wiring where useful.
Validate with the focused test file and `uv run marimo check
notebooks/analysis_app.py`.
