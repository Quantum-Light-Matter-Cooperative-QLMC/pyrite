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

- [x] Reproduce with a synthetic record whose line grid is much finer than its
      wide-brem grid and whose combined frame exceeds the chart point budget.
      Compare total-trace energy coordinates, median/max spacing near the line,
      peak height, and FWHM with background disabled versus enabled.
- [x] Confirm the user's representative checkpoint shows the same mechanism;
      record source grid sizes, rendered row counts, and local line-region
      spacing. Do not infer stored-spectrum corruption from visual appearance.
- [x] Freeze the externally meaningful invariant: toggling the background may
      add a continuum trace/tail but must not coarsen or broaden the displayed
      coherent-line contribution over the shared line-energy domain.
- [x] Adjust plot-time data preparation/decimation so smooth background rows do
      not consume the line trace's required resolution. Preserve the global
      Vega-Lite payload bound or replace it with an explicit, documented
      per-trace/per-band budget.
- [x] Cover `spectrum_frame`/`spectrum_chart` with background on and off,
      multiple beam energies, narrow and broad views, and a wide continuum;
      assert energy-grid and line-shape invariants rather than only chart
      serialization.
- [x] Check polar-angle, azimuthal, energy-comparison, and pixel-selected
      callers for the same chart contract; add coverage only where behavior
      differs.
- [x] Run focused tests, apps suite, `marimo check
      notebooks/analysis_app.py`, a real no-browser `cxr app analysis --smoke`,
      lint/type checks, and scoped diff review.

## Decisions and open questions

- Preserve fine-line resolution is the primary contract; the background trace
  is subordinate and may be decimated more aggressively because it is smooth.
- Keep interpolation direction unchanged unless reproduction disproves current
  code evidence: coarse `brem_wide` is interpolated onto fine `E_grid` for the
  line region, while the broad tail remains on `E_grid_brem` after the line
  grid ends.

Resolved during implementation:

- Two plot-time effects caused the loss: total and brem components split the
  point budget equally, then the wide tail competed with the fine line grid.
  No line spectrum was interpolated onto `E_grid_brem`; stored checkpoint data
  remained correct.
- `max_points` now bounds distinct serialized energy-coordinate rows across
  logical traces (minimum three per trace). Total and brem values share those
  rows through a Vega-Lite fold transform. The full line grid is retained when
  it fits; otherwise it receives 90% of its trace budget and the wide tail 10%.
- Detector convolution is unrelated: the reproduced defect occurs with
  detector QE and convolution disabled, inside intrinsic-spectrum decimation.

## Implementation evidence

- Synthetic 12,000-point line / 1,200-point wide-brem record at
  `max_points=500`:
  - before: background off/on total rows 500/250; line-window samples 26/11;
    max local spacing 8.0007/35.0029 eV;
  - after: background off/on total rows 500/500; line-window samples 26/24;
    max local spacing 8.0007/8.6674 eV; peak coordinate, peak height, and
    sampled FWHM unchanged.
- Representative local HOPG checkpoint: 3,332 records; selected 300 keV case
  has 3,018 line-grid and 12,001 wide-brem points. Background off/on both retain
  all 67 samples in the measured peak window with identical 2.9997 eV maximum
  spacing; all 3,018 line-grid coordinates remain exact, while the remaining
  serialized budget carries the wide tail.
- Focused Altair spectrum tests: 35 passed. Apps suite: 287 passed.
- `cxr-dev lint`, `cxr-dev typecheck`, `marimo check
  notebooks/analysis_app.py`, `cxr app analysis hopg --smoke`, and
  `git diff --check` passed. Sandboxed Marimo checks could not open a local
  multiprocessing listener or terminate normally; approved outside-sandbox
  reruns passed.

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
