# Analysis spectrum resolution with bremsstrahlung

Branch: `fix/analysis-spectrum-resolution-with-brem`
TODO scope: direct user report from 2026-08-01.

## Problem

In `cxr app analysis`, enabling **show brem background** makes energy spectra
look coarser, as though the fine coherent-line grid were interpolated onto the
coarse bremsstrahlung grid instead of the continuum being interpolated onto
the line grid. The displayed line width/bin spacing must not change merely
because the background trace is visible.

Current checkpoint production already stores `brem` interpolated onto the fine
`E_grid`, and `src/cxr_mc/plots/altair_spectra.py::_record_frame` builds the
line-region total on that grid. A stronger initial suspect is plot-time
decimation: `spectrum_chart` defaults to `max_points=5000`, while
`_decimate_frame` divides that budget across all trace groups. Enabling brem
adds a `component="brem"` group and can reduce the samples retained for the
`component="total"` line trace. Reproduction must distinguish this from actual
grid interpolation, browser/Vega-Lite rendering, or detector convolution
before changing behavior.

## Scope and owners

- `src/cxr_mc/plots/altair_spectra.py`: spectrum-frame composition,
  per-component plot-point budgeting, and peak/line-shape preservation.
- `tests/test_altair_plots.py` and `tests/test_altair_spectra_compare.py`:
  focused synthetic regressions for energy coordinates, line width, and
  background-enabled/disabled parity.
- `notebooks/analysis_app.py`: caller defaults or controls only if the reusable
  chart contract requires them; keep the marimo app thin.
- `src/cxr_mc/plots/altair_detectors.py` only if reproduction shows the same
  defect in detector-response spectrum views.

Out of scope: changing scan grids, recomputing checkpoints, modifying physical
line or bremsstrahlung kernels, increasing Monte Carlo sample counts, or
redesigning unrelated analysis tabs.

## Implementation checklist

- [ ] Reproduce with a synthetic record whose line grid is much finer than its
      wide-brem grid and whose combined frame exceeds the chart point budget.
      Compare total-trace energy coordinates, median/max spacing near the line,
      peak height, and FWHM with background disabled versus enabled.
- [ ] Confirm the user's representative checkpoint shows the same mechanism;
      record source grid sizes, rendered row counts, and local line-region
      spacing. Do not infer stored-spectrum corruption from visual appearance.
- [ ] Freeze the externally meaningful invariant: toggling the background may
      add a continuum trace/tail but must not coarsen or broaden the displayed
      coherent-line contribution over the shared line-energy domain.
- [ ] Adjust plot-time data preparation/decimation so smooth background rows do
      not consume the line trace's required resolution. Preserve the global
      Vega-Lite payload bound or replace it with an explicit, documented
      per-trace/per-band budget.
- [ ] Cover `spectrum_frame`/`spectrum_chart` with background on and off,
      multiple beam energies, narrow and broad views, and a wide continuum;
      assert energy-grid and line-shape invariants rather than only chart
      serialization.
- [ ] Check polar-angle, azimuthal, energy-comparison, and pixel-selected
      callers for the same chart contract; add coverage only where behavior
      differs.
- [ ] Run focused tests, apps suite, `marimo check
      notebooks/analysis_app.py`, a real no-browser `cxr app analysis --smoke`,
      lint/type checks, and scoped diff review.

## Decisions and open questions

- Preserve fine-line resolution is the primary contract; the background trace
  is subordinate and may be decimated more aggressively because it is smooth.
- Keep interpolation direction unchanged unless reproduction disproves current
  code evidence: coarse `brem_wide` is interpolated onto fine `E_grid` for the
  line region, while the broad tail remains on `E_grid_brem` after the line
  grid ends.
- Open: is apparent resolution loss entirely `_decimate_frame` budgeting, or
  does Altair/Vega-Lite apply another transform after serialized data creation?
- Open: should `max_points` mean a whole-chart cap, a per-component cap, or a
  weighted budget that reserves line-region samples? Choose from measured
  payload/render cost while keeping an explicit upper bound.
- Open: does detector convolution intentionally set final spectral resolution
  in any affected tab? If so, keep that physical broadening distinct from this
  background-toggle invariant.

## Delegation

One normal `implement-task` slice: reproduce, add the failing renderer-level
regression, make the smallest shared plotting fix, then exercise all analysis
callers. Required skills: `notebook-workflow`, `regression-testing`, and
`scientific-library`; use `run-cxr-mc` for the real app smoke path. No heavy
Monte Carlo, remote GPU work, or physics-validation task is required unless
reproduction exposes stored numerical corruption.

## Acceptance checks

- With identical records and chart point budget, enabling brem leaves the
  rendered total trace's fine-grid coordinates and line-region sampling no
  coarser than the background-disabled trace, or preserves line peak/FWHM to a
  documented tolerance justified by the decimator.
- The background trace remains present and follows `E_grid` through the line
  region; the wide tail reaches the valid beam-energy endpoint on
  `E_grid_brem` without resampling the line onto that coarse grid.
- Serialized chart payload stays within an explicit bounded point budget and
  retains peak-preserving behavior for multiple beam energies/components.
- Focused Altair spectrum tests pass for narrow, broad, comparison, and
  pixel-selected-equivalent inputs.
- `marimo check notebooks/analysis_app.py` and `cxr app analysis --smoke`
  succeed through the project runner; relevant apps tests, lint/type checks,
  and `git diff --check` pass.
