# Plan: "Compare any cases" tab in `analysis_app.py`

Status: **partially implemented** (2026-07-26). Steps 1-3 done, lint/tests/
`marimo check` green. Step 4 (metrics panel) and step 5 (docs/TODO/verify)
NOT started.

## WIP handoff

Done: `selection.py` (`case_label`, `case_table_rows`, `slim_case_record`),
`altair_spectra.py` (`multi_case_spectrum_chart` + `_multi_case_frame`),
exports synced (`results/__init__.py`, `test_results_exports.py`), tests
green (`test_results_selection.py` 14, `test_altair_spectra_compare.py` 17
incl. `multi_case_spectrum_chart` cases). Notebook wiring in
`notebooks/analysis_app.py`: basket state cell (`mo.state`), picker cell
(`case_picker_ui`), add/remove/clear button cells, narrow+broad spectral
control cells (mirroring `polar_compare_tab`), and `case_compare_tab()`
registered in the `Explore` accordion as "Compare any cases". `uvx marimo
check` clean; `test_analysis_app.py` 14 pass (the remove-select widget needed
its own cell, split from the buttons that read its `.value` — a cell can't
both create a `mo.ui` widget and read its own `.value`, per
`test_all_ui_values_are_read_downstream_of_creation`). `dev.py lint` clean.

Next: step 4 (per-basket-entry metrics table: peak flux, line flux,
prominence, line-to-brem ratio) then step 5 (docs/TODO/verify). See
"Implementation order" below.

## Goal

A new analysis-app view where the user hand-picks *individual cases* — any
combination of beam energy, crystal thickness, polar tilt, and azimuth — and
overlays their spectra and performance metrics. Selection must work **across
materials and faces**, not just within the currently loaded checkpoint.

## Why the existing pieces don't cover this

- Each existing compare tab (`spectra_tab`, `polar_compare_tab`,
  `azim_compare_tab`) varies exactly **one** case field with the others pinned
  by dedicated dropdowns.
- `compare_spectrum_chart` / `_compare_frame`
  (`src/cxr_mc/plots/altair_spectra.py:347`) hue by a single case field and
  *collapse* duplicate records sharing a hue value to the strongest-peak
  record — wrong for arbitrary case-vs-case overlays, where two records may
  legitimately share every hue-able field.
- The `Compare` (cross-material) tab reduces each material to one summary
  point via `cached_analysis` + `material_comparison_point`; it never exposes
  individual spectra.
- The app loads **one** material+face checkpoint at a time
  (`load_analysis_checkpoint`, `notebooks/analysis_app.py:158-191`).
  Checkpoints are 120–240 MB pickles (see `checkpoints/`), so naive
  multi-checkpoint loading is a memory hazard, especially under WSL.

## Design

### Cross-material strategy: "case basket", not multi-checkpoint loading

Rather than holding N full checkpoints in memory, reuse the existing
single-material loader as the *browser* and accumulate selections in a
lightweight persistent store:

1. User browses cases of the currently loaded material/face (existing
   top-of-app `material_ui` / `face_ui` selectors).
2. A case-picker table lists every record in the loaded checkpoint.
3. An **"Add to comparison"** button appends *slimmed copies* of the selected
   records to a `mo.state`-backed basket, tagged with material key, catalog
   label, and face.
4. Switching material via the normal selector and adding more rows achieves
   cross-material comparison — no second checkpoint is ever resident; only the
   basket (a handful of spectra, ~kB–MB each) persists.

Basket entries survive material switches by construction (`mo.state` is
independent of the `res` dependency chain). Provide remove-row and clear-all
controls. Cap basket size (~12 curves) with an explicit truncation notice —
no silent drops.

Rejected alternatives:

- *Load N checkpoints keyed by a material multiselect*: memory-unsafe
  (multiple 200 MB pickles), duplicates loader logic, slow first paint.
- *Extend `.meta.json` manifests with full case lists to build a global
  picker without loading pickles*: still requires loading the pickle to get
  spectra, so it only optimizes browsing; can be a later enhancement if
  browsing-before-loading is wanted.

### 1. Library additions (logic in `src/cxr_mc/`, never in the notebook)

**a. `src/cxr_mc/results/selection.py` — case identity + slimming**

- `case_label(case, *, material_label=None, face=None, varying=None)` —
  human-readable label, e.g. `"HOPG · 100 keV · 5 µm · tilt −80° · az 120°"`.
  `varying` (output of `sweep_values`) elides fields that only take one value
  within the basket; material and energy always shown when present.
- `case_table_rows(results)` — one plain dict per record for the picker
  table: `name`, `E0_keV`, `thickness_ang`, `tilt_deg`, `tilt_azim_deg`, plus
  cheap metrics (peak flux via `_peak`; line flux / prominence if computable
  without heavy analysis). Row order gives a stable row→record mapping.
- `slim_case_record(record, *, material, label, face)` — minimal dict a
  basket entry needs to plot later: `case`, spectrum array(s), energy grid,
  and whatever else `_record_frame` (`plots/altair_spectra.py:154`) touches
  (verify exact field set during implementation; pattern after
  `slim_results`). Everything else from the checkpoint record is dropped so
  the basket stays small.

**b. `src/cxr_mc/plots/altair_spectra.py` — `multi_case_spectrum_chart`**

- Signature mirrors `compare_spectrum_chart` (`include_brem`, `x_domain`,
  `x_type`, `y_type`, `band`, `max_points`, `width`, `height`) but takes an
  explicit list of (record, label) pairs.
- **One line per record, no best-peak collapse** — the key difference from
  `_compare_frame`.
- Color channel = the composite label (nominal). Reuse `_record_frame`,
  `_scale`, `_log_y_scale` / `_linear_y_scale`, `_decimate_frame`, and the
  dashed-brem layer pattern.
- Returns `None` for an empty input; raises nothing on mixed materials.

**c. Performance panel**

- Per-basket-entry metrics table: peak flux, line flux, prominence,
  line-to-bremsstrahlung ratio — reusing the metric extraction behind
  `plots/sweeps` (`_HEATMAP_QUANTITIES`, `metric_vs_chart` internals).
- Start as a plain `mo.ui.table` (exact numbers, cheapest); a grouped bar
  chart per metric is an optional follow-up.

### 2. Notebook wiring (`notebooks/analysis_app.py`)

All widgets in top-level cells (marimo reactivity requirement, matching the
existing tabs' comments).

- **Basket state cell**: `basket, set_basket = mo.state([])` plus add /
  remove / clear callbacks. Entries are `slim_case_record` outputs.
- **Picker cell**: `mo.ui.table(case_table_rows(res), selection="multi")`
  over the RAW `res` (like `polar_compare_tab` — thickness is a selectable
  dimension here, so the top-of-notebook thickness pin must not apply),
  followed by an `mo.ui.button` "Add selected cases" whose handler slims the
  selected rows into the basket with the current `MATERIAL` / `FACE` tags.
- **Controls cell**: brem toggle, x/y log, narrow/broad band, axis bounds —
  mirror the `polar_compare_tab` control set with independent widget
  instances.
- **Tab function cell** `case_compare_tab()`:
  - empty basket → `mo.md` hint ("add 2+ cases from the picker");
  - otherwise `multi_case_spectrum_chart` over the basket + metrics table +
    a basket-contents table with per-row remove.
  - Blazed-face entries get their `(blazed)` marking via the label (the
    per-chart `face_title` wrapper can't distinguish mixed-face baskets — the
    label carries the face instead).
- **Register** in the tab dict (`notebooks/analysis_app.py:1741`): entry in
  the `Explore` accordion, `"Compare any cases": case_compare_tab`. Accordion,
  not nested `mo.ui.tabs` — nested tabs silently blank charts
  (marimo-team/marimo#6919, see `detectors_tab` comment).

### 3. Tests

- `case_label`: formatting, varying-field elision, material/face tagging.
- `case_table_rows`: row↔record mapping, metric columns present.
- `slim_case_record`: output is plottable by `_record_frame`; size sanity
  (no full-checkpoint payloads leak into the basket).
- `multi_case_spectrum_chart`: empty → `None`; single record; N records give
  N distinct color values; two records sharing every hue-able field both
  survive (regression against `_compare_frame`-style collapse); brem layer
  toggles; mixed-material labels render.
- Notebook-level tests follow existing `tests/test_analysis_app.py`
  patterns; run `uvx marimo check notebooks/analysis_app.py` and fix all
  findings.

### 4. Docs / hygiene

- `docs/repo_map.md` and README notebook section: one-liner for the new tab.
- `TODO.md` on `main`: add/check the tracked item before completion
  (`todo-sync` conventions).
- Full gate: `scripts/dev.py verify` via the canonical `rtk env ... uv run`
  invocation.

## Implementation order

1. `selection.py` helpers (`case_label`, `case_table_rows`,
   `slim_case_record`) + unit tests.
2. `multi_case_spectrum_chart` + unit tests.
3. Notebook cells: basket state → picker → controls → tab function →
   registration; `uvx marimo check`.
4. Metrics panel.
5. Docs, TODO sync, `dev.py verify`.

## Open questions

- Basket persistence across app restarts (serialize to a small JSON next to
  checkpoints?) — out of scope for v1; basket is session-local.
- Whether the picker table should surface heavier quality metrics
  (line-definition quality) — depends on their compute cost per record;
  decide during implementation.
- Extend `.meta.json` manifests with case inventories to allow browsing an
  unloaded material's cases — deferred enhancement, see rejected
  alternatives.
