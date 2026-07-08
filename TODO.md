# TODO (branch: feature/marimo-altair-followups)

Scope: `main`'s P3 #1 "Marimo/Altair follow-ups" -- the remaining notebook/plot
fixes after the core `feature/marimo-transfer` migration landed.

## Done on this branch

- **Fix blank detector-tab plots.** Root cause: `analysis_app.py`'s Detectors tab
  nested `mo.ui.tabs(...)` (Eagle XO / Timepix3) inside the outer `mo.ui.tabs`
  (lazy=True) -- a known marimo bug where charts inside nested tabs render blank
  (marimo-team/marimo#6919). Fixed by rendering the Eagle XO / Timepix3 split as
  a `mo.accordion` instead (same lazy-defer behavior, no nesting problem).
- **Fix penetration plots for multilayer/stacked materials (mos2).** Two bugs:
  (1) `config.trajectory_sweep()` never propagated a material's `substrate`/
  `stack` into the `Sweep` it built, so the Penetration tab's cases always had
  `abs_layers=None` even for mos2 (substrate="sapphire") -- fixed by carrying
  `substrate`/`stack` through. (2) `plots/trajectories._trajectory_data` (the
  single choke point for both the matplotlib and Altair trajectory/survival
  renderers) called `simulate_trajectories` with only the film's
  `composition`/`thickness_ang`, ignoring `case["abs_layers"]` -- fixed to pass
  `layers=case.get("abs_layers")` and use the full stack's total thickness for
  the depth axis. Internal layer-boundary lines now drawn on both renderers.
- **Angle selector for penetration plots.** Added a polar-tilt dropdown
  (`penetration_angle_ui`) driving both the survival-vs-depth chart and the
  single-track view; the geometry sweep now uses 15 deg spacing
  (`tilt_span=75, n_tilts=11`) so the default (15 deg) lands exactly on the grid.
- **Intrinsic spectra without brem + adjustable x-limits + lin/log y.**
  `altair_spectra.spectrum_chart` already supported `include_brem=False`; added
  `x_domain=`/`y_type=` params and wired a brem-toggle checkbox, x-min/x-max
  number inputs, and a log-y switch in the Intrinsic-spectra tab. The same
  `x_domain` also threads through to `timepix_detected_chart` /
  `eaglexo_detected_chart` (the other two spectral, energy-vs-intensity views).
- **Finalize transition to `scan_app.py` (material selection).** Was hardcoded
  `MATERIAL = "hopg"`; replaced with a `mo.ui.dropdown` over `config.MATERIALS`.
  Also dropped the leftover `IPython.display` import/call (marimo renders a
  cell's bare last expression directly).
- **Fixed heatmap plot size + few-decimal tick labels.** `altair_sweeps.heatmap_chart`
  sized each panel as `cell * n_distinct_values`, so a dense sweep (e.g. a
  25/40-tilt grid) grew the chart rather than shrinking the cells -- switched to
  a fixed `width`/`height` per panel (Vega-Lite auto-divides the ordinal range).
  Added `axis=alt.Axis(format=".3~g")` to the heatmap x/y and `metric_vs_chart`
  axes so float sweep values (e.g. `-33.300000000000004`) render as short
  tick labels.

## Deferred / follow-ups spotted but out of scope

- `eaglexo_charge_chart` doesn't yet take `x_domain=` (only the two
  detected-vs-incident charts do).
- The dense penetration-grid accordion (matplotlib, in the Penetration tab) is
  still fixed at `energy=30` regardless of the angle selector above it.
