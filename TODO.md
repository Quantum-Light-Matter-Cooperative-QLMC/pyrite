# TODO — refactor/dedup-followthrough

This branch carries one backlog item; the full triaged backlog lives on `main`.

## Codebase de-duplication follow-through (P2 #4 — usability/maintainability)

`/sc:analyze` (2026-07-02) inventoried 7 MEDIUM duplication clusters across the
plots/detector/sweep layers. **Done already** (on `main`, commits `e6edc50` + `6f51ced`):
the HIGH correctness bug (`repair_brem_wide` multilayer drift), the LOW dead-code nits,
the best-azimuth collapse idiom (`_best_azimuth`/`_peak_line` in `plots/_common.py`,
~11 sites), and the `cases→names` prologue (`results.records_for_cases`, 9 sites).
**Also done** (this branch): the M7 remainder — `_case_title` and `_metrics_map` hoisted
into `plots/_common.py`, 8 and 5 call sites swapped respectively (verified with a
scratch literal-vs-helper diff against a representative case, since no test asserts
title text; one intentional cosmetic LaTeX shift at `detectors.py`'s Eagle
charge-density title — equals sign moved inside `\theta_\mathrm{tilt}` math mode for
consistency, spacing only). Left alone: `interactive.py`'s plotly slider `_title`
(distinct compact unicode label, not a byte duplicate) and `spectra.py`'s mosaic
title (uses `$E_0$=` notation, a genuine format divergence, not the same recipe).
**Also done** (this branch): M2 — Si constants + `grid_key`/`prep_spectrum`/
`poisson_core` hoisted into a new `_si_sensor.py`, imported by both
`timepix_response.py` and `eaglexo_response.py` (no ledger entries needed — verbatim
move, no pre-existing `Validation:` markers). M3 — `plot_heatmaps` gained a
`value=callable(record)` mode (factored into `sweeps._value_heatmap`, with its own
`auto_lines` thin-axis fallback); `plot_eaglexo_charge_map` is now a ~25-line wrapper
around it, smoke-tested against the `wse2` checkpoint (heatmap, `exposure_s`, and
thin-axis line-plot paths). **Also done** (this branch): M5 — `sweep.crystal_params`'s
110-line if-chain collapsed into a `_CRYSTAL_PARAMS` registry (a `TypedDict` table
mirroring `config._MATERIAL_GRIDS`), with `composition` now derived via
`substrate_composition(material)` (reusing its existing `Counter`-over-`basis` logic)
instead of hand-listed per material; HOPG's fixed `hkl_list` (fiber-textured, skips
the automatic `dominant_reflections` family search) is the one registry override.
`substrate_radiator`, `layer_radiator`, and the inline film-radiator dict in
`build_cases` now all build off one `_radiator(cp, beam_uvw=, azimuth_rad=)`
constructor. `sweep.py` 550 -> 520 lines net (registry table + its TypedDict/comments
add lines back; the dead `n_of` helper, now unused, was also deleted). **Also done**
(this branch): the `results.py` package split — 663 lines split verbatim into a
`results/` package (`store.py`: `Settings`/`store_result`/`detected_background`;
`selection.py`: `records`/filter/select/slim helpers/`best_azimuth`; `metrics.py`:
`line_metrics` and its peak-finding helpers; `scoring.py`: `selection_score`/
`top_geometries`/`show_top`; `tables.py`: `results_dataframe`/`summary_table`), behind
a new `tests/test_results_exports.py` mirroring the `montecarlo`/`plots` export-freeze
pattern. Verified against the pre-split module by an AST source-diff of every moved
function/constant (byte-identical bodies; the one AST "diff" was a decorator-boundary
artifact of `inspect.getsource`, not a content change) plus the full suite. **Also done**
(this branch): M6 — the renderer-neutral frame builders `heatmap_frame`/`metric_vs_frame`/
`_effective_x`/`_ndistinct`/`scan_mode`/`pick_hue` moved into a new `plots/_frames.py`;
`sweeps.py`'s `plot_heatmaps`/`plot_metric_vs`/`plot_scan` now render from those same
functions instead of re-deriving the per-cell/per-point best-record reduction inline, so
matplotlib and Altair (`altair_sweeps.py`) share one reduction instead of two. Verified
byte-identical against the pre-refactor code via a scratch before/after diff script over
synthetic records (mesh/line data unchanged) plus the full suite. This closes the branch's
numbered TODO backlog (M4 and the M7 `line_fwhm_eV`/escape-helper sub-items remain tracked
only in `docs/dedup-inventory.md`, out of this branch's scope).

Full original inventory + rationale (tracked, self-contained — includes M4 and the M7
`line_fwhm_eV`/escape-helper sub-items not summarized below):
[`docs/dedup-inventory.md`](docs/dedup-inventory.md).

**Hard constraints on every item below:**
- `tests/test_plots_exports.py` + `tests/test_montecarlo_exports.py` +
  `tests/test_results_exports.py` freeze every re-exported name
  (`set(pkg.__all__) == FROZEN_EXPORTS`). Adding names is fine; removing
  or failing to re-export breaks the test. `altair_*` modules are intentionally NOT
  re-exported — don't add them.
- Anything touching `montecarlo/` or `crystallography.py` triggers the physics-validation-ledger
  workflow (derivation docstring + `Validation:` marker + ledger row + fresh-context
  verification per `docs/validation/README.md`). Prefer verbatim moves over rewrites there.
- Verify every step with `uv run pytest`, `uv run ruff check .`, `uv run pyright` — all
  three must stay clean (198 tests / 0 ruff / 0 pyright as of branch start).

### Remaining work, in suggested order

This branch's numbered TODO backlog is now empty — M6 (the last item) is done above. M4
and the M7 `line_fwhm_eV`/escape-helper sub-items remain tracked only in
`docs/dedup-inventory.md`, out of this branch's scope.

### Notes carried from the analysis session
- `plots/detectors.py` (677 → 599 lines after M1+M3) is the best large-file refactor
  target left; what's left splits cleanly at the existing `# ---- Timepix` /
  `# ---- Eagle XO` comment seams.
- Leave `plots/trajectories.py` (530, cohesive) and `checks/feranchuk_spence.py`
  (1056, standalone validation anchor) alone — not part of this cleanup.
- `analysis.py` (matplotlib, 453) vs `notebooks/analysis_app.py` (altair, 355) are
  parallel marimo apps; out of scope here, but worth a retirement date for the
  matplotlib one once altair parity is confirmed.
