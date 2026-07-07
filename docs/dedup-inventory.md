# Codebase de-duplication inventory (P2 #4)

Self-contained record of the `/sc:analyze` duplication audit run 2026-07-02 and the
follow-through work on the `refactor/dedup-followthrough` branch. This doc exists so the
inventory survives independently of the local `CC-Session-Logs/` (which is gitignored and
does not travel between machines). `TODO.md` on this branch carries the *suggested order of
attack*; this file carries the *full findings with line numbers and rationale*.

## Hard constraints on every item

- **Frozen exports.** `tests/test_plots_exports.py` + `tests/test_montecarlo_exports.py`
  assert `set(pkg.__all__) == FROZEN_EXPORTS`. Adding names is fine; removing or
  failing-to-re-export a frozen name breaks the test. `altair_*` modules are intentionally
  **not** re-exported — don't add them.
- **Physics-validation ledger.** Anything touching `montecarlo/` or `crystallography.py`
  triggers the ledger workflow (derivation docstring + `Validation:` marker + ledger row +
  fresh-context verification per `docs/validation/README.md`). Prefer **verbatim moves over
  rewrites** there.
- **Verification baseline.** Every step must keep `uv run pytest`, `uv run ruff check .`,
  and `uv run pyright` clean. Baseline at branch start: 198 tests / 0 ruff / 0 pyright.

## Status

**Done and committed:**

| Item | What | Commit |
|------|------|--------|
| H1   | `repair_brem_wide` multilayer drift (correctness) routed through the multilayer runner path | `e6edc50` |
| LOW nits | dead-expr / dead-statement / redundant-import one-liners | `e6edc50` |
| M1   | best-azimuth collapse idiom → `_best_azimuth`/`_peak_line` in `plots/_common.py` (~11 sites) | `6f51ced` |
| M7 (prologue) | `cases→names` prologue → `results.records_for_cases` (9 sites) | `6f51ced` |
| M7 (remainder) | `_case_title`/`_metrics_map` hoisted into `plots/_common.py` (8 + 5 sites) | `f6f0dd2` |
| M2   | Si constants + `grid_key`/`prep_spectrum`/`poisson_core` hoisted into `_si_sensor.py` | `28e20ec` |
| M3   | `plot_eaglexo_charge_map` rewritten as a wrapper around `plot_heatmaps`' new `value=` mode | `28e20ec` |
| M5   | `sweep.crystal_params` if-chain → `_CRYSTAL_PARAMS` registry; radiator-dict constructions unified into `_radiator()` | `546fd5c` |
| —    | `results.py` package split (`store`/`selection`/`metrics`/`scoring`/`tables` + `test_results_exports.py`) | `279e709` |
| M6   | `heatmap_frame`/`metric_vs_frame`/`_effective_x`/`_ndistinct`/`scan_mode`/`pick_hue` hoisted into `plots/_frames.py`; `sweeps.py` renders from them | `83e16b9` |

**Remaining:** nothing queued in `TODO.md`'s numbered backlog. **Two items that fell out of
the TODO summary and must not be lost: M4, and the M7 `line_fwhm_eV`
/ escape-helper sub-items — see below.**

---

## Findings inventory

### HIGH — H1: `run.repair_brem_wide` multilayer drift  *(DONE — `e6edc50`)*

`run.py:275-284` re-implemented the runner's brem phase by hand instead of reusing
`runner._transport_case`/`_spectrum_case`, and had drifted: it called
`simulate_trajectories(...)` and `mc_brem_spectrum(..., n_hat=n_hat)` **without
`layers=case["abs_layers"]`** and without the per-layer brem sum (`runner.py:178-190`).
Repairing a stacked/multilayer checkpoint silently regenerated **single-slab** brem (no
substrate backscatter/brem/cross-stack absorption) and wrote it back. Also ignored
`brem_chunk`. Fixed by routing the repair through the runner's brem path.

### MEDIUM — duplication clusters

- **M1 — collapse idiom ×10-11.**  *(DONE — `6f51ced`)*
  `max(grp, key=lambda r: float(np.max(r["spec"])))` + group-by-energy loop across
  `spectra.py` ×3, `detectors.py` ×4, `interactive.py` ×2, `altair_spectra.py`,
  `altair_detectors.py`. Hoisted `_peak_line(r)` and `_best_azimuth(grp, collapse_azimuth)`
  into `plots/_common.py` (added `import numpy as np`). Internal helpers, not re-exported.

- **M2 — timepix vs eaglexo response share verbatim blocks.**  *(DONE — `28e20ec`.)*
  `timepix_response.py` and `eaglexo_response.py` duplicated: the Si constants
  (`W_EHP_EV`, `FANO_SI`, `SI_DENSITY_G_CM3`, `SI_A`, `SI_N_PER_ANG3`), the
  `_RESPONSE_CACHE` + `get_response` grid-key caching pattern, the `poisson_counts` core,
  and the `apply()` shape-guard text. Extracted a new `_si_sensor.py` holding the constants +
  `grid_key(E)` + `prep_spectrum(spec, E_grid, module_name)` (the shape guard) +
  `poisson_core(E_grid, detected_per_s, time_s, rng)`; both modules import the constants and
  delegate to the three helpers, keeping their own physics (charge-sharing MC, QE table, ...)
  untouched. No ledger entries needed — verbatim relocation, no existing `Validation:` markers
  to preserve (both ledger rows were already `unverified`/`blocked` pre-refactor), confirmed
  against `docs/physics-validation-ledger.md` before assuming so.

- **M3 — `plot_eaglexo_charge_map` re-implements `plot_heatmaps`.**  *(DONE — `28e20ec`.)*
  `plots/detectors.py:562-677` duplicated ~90 lines of `sweeps.py` best-per-cell /
  `_cell_edges` / shared vmin-vmax / colorbar machinery, plus the thin-axis→line fallback. Gave
  `plot_heatmaps` a `value=callable(record)` mode (factored into a new `sweeps._value_heatmap`
  helper: max-reduces `value` per cell instead of going through line_metrics/`selection_score`,
  with its own `auto_lines` thin-axis → line-plot fallback since `plot_heatmaps` callers other
  than `plot_scan` don't get that decision for free). `plot_eaglexo_charge_map` is now a
  ~25-line wrapper building the `_val`/`label` closure and delegating; smoke-tested against the
  `wse2` checkpoint (heatmap mode, `exposure_s` well-fill mode, and the thin-axis line-plot
  fallback via `x="thickness_ang"`, which is single-valued in that checkpoint).

- **M4 — wide-brem overlay physics ×4.**  *(TODO — NOT in the TODO summary; preserved here.)*
  `inc_b = brem_wide*scale; det_b = inc_b*qe(...)` (plus the charge variant `*Eb/W_EHP_EV`)
  reappears in `_draw_eaglexo_detected`, `_draw_eaglexo_charge`, `eaglexo_detected_frame`,
  and `eaglexo_charge_frame`. Extract `_eag_wide_brem(r, coating)` / `_eag_wide_charge(r, coating)`
  next to `_eag_detected`.

- **M5 — `sweep.crystal_params` 110-line if-chain.**  *(DONE — `546fd5c`.)*
  Every branch returned `{crystal, composition, hkl_list, beam_uvw, B_ang2, E_grid}`. Collapsed
  to `_CRYSTAL_PARAMS`, a data registry mirroring `config._MATERIAL_GRIDS` (a `CrystalParamsGrid`
  `TypedDict` table keyed by material, holding just `B_ang2`/`beam_uvw`/`E_grid` plus an optional
  `hkl_list` override); `crystal` and `composition` follow mechanically from the key
  (`composition=substrate_composition(material)`, reusing its existing `Counter`-over-`basis`
  logic instead of hand-listing per material — verified the basis order in
  `crystal_structures.toml` puts the metal/majority element first for every material, matching
  the old hand-written tuples). HOPG keeps its fixed `hkl_list` (fiber-textured — skips the
  automatic `dominant_reflections` family search) as the registry's one override; every other
  material derives `hkl_list` from `dominant_reflections(material, n_families, B_ang2)` as
  before. Also unified the three radiator-dict constructions (`substrate_radiator`,
  `layer_radiator`, the inline film dict at old `sweep.py:472-479`) into one `_radiator(cp, *,
  beam_uvw=None, azimuth_rad=None)` constructor — `azimuth_rad` is only added to the dict when
  given, preserving `substrate_radiator`'s narrower 4-key contract
  (`test_substrate_radiator_crystalline_vs_amorphous` asserts the exact key set). Deleted the
  now-dead `n_of` helper (only caller was the removed hand-listed composition tuples).
  `sweep.py` went 550 → 520 lines net (the registry table + its `TypedDict`/comments add lines
  back against the if-chain's removal). 198 tests / 0 ruff / 0 pyright unchanged.

- **`results.py` package split.**  *(DONE — `279e709`.)* 663 lines split
  verbatim (no logic changes) into a `results/` package following the `montecarlo`/`plots`
  precedent: `store.py` (`Settings`, `store_result`, `detected_background`), `selection.py`
  (`records`/`records_for_cases`/`filter_results`/`sweep_values`/`select_results`/
  `slim_results`/`best_azimuth`), `metrics.py` (`line_metrics` + its peak-finding helpers),
  `scoring.py` (`selection_score`/`top_geometries`/`show_top`), `tables.py`
  (`results_dataframe`/`summary_table`/`show_summary`); `__init__.py` re-exports every name
  (public and internal) the old module defined. New `tests/test_results_exports.py` freezes
  the set, mirroring `test_montecarlo_exports.py`/`test_plots_exports.py`. Verified against
  the pre-split module with an AST-based source diff of every moved function/constant
  (byte-identical; the sole flagged "diff" was `inspect.getsource` including the `@dataclass`
  decorator line that the AST body-segment comparison for the old class didn't capture — a
  comparison artifact, not a content change) plus the full 201-test suite / 0 ruff / 0 pyright.

- **M6 — matplotlib `sweeps` vs `altair_sweeps` duplicate the reduction, not just rendering.**
  *(DONE -- `83e16b9`.)* Moved `heatmap_frame`/`metric_vs_frame`/
  `_effective_x`/`_ndistinct`/`scan_mode`/`pick_hue` into a new renderer-neutral
  `plots/_frames.py`. `sweeps.py`'s `plot_heatmaps`/`plot_metric_vs`/`plot_scan`
  now render from the same tidy-data frames `altair_sweeps.py`'s
  `heatmap_chart`/`metric_vs_chart`/`scan_charts` already did -- both renderers
  share one reduction. `_AXIS_SPECS`/`_axis_disp`/`_value_label`/`_FLUX_GATED`
  moved from `sweeps.py` into `_frames.py` (sweeps.py imports them back) to
  keep the `_frames -> sweeps` dependency acyclic. Verified via a scratch
  before/after diff script (matplotlib line/mesh data byte-identical) plus
  the full suite.

- **M7 — smaller mechanical hoists.**
  - `records_for_cases(results, cases)` — the `cases→names` prologue ×9. *(DONE — `6f51ced`.)*
  - `_case_title` — extract from `altair_detectors._title(...)`; title recipe reimplemented
    inline ~12× across `spectra.py`, `detectors.py`, `sweeps.py`, `altair_sweeps.py`,
    `interactive.py`. Target: `plots/_common.py` next to `_best_azimuth`/`_peak_line`.
    *(DONE — `f6f0dd2`; 8 call sites swapped. `sweeps.py` had no case-title
    code, so only the other 4 files needed edits. Left alone: `interactive.py`'s plotly
    slider title and `spectra.py`'s mosaic title — genuine format divergences, not the same
    recipe.)*
  - `_metrics_map` — extract from `altair_sweeps._metrics_map(...)`; reimplemented ×5. Same
    target module. *(DONE — `f6f0dd2`; 5 call sites swapped across
    `altair_sweeps.py`, `spectra.py`, `sweeps.py`.)*
  - `line_fwhm_eV(case, E_pk, mosaic_rad)` — `store_result` (`results.py:64-80`) and
    `plot_mosaic_comparison` (`spectra.py:254-275`) duplicate the EDS² + aperture² +
    capped-mosaic quadrature. *(TODO — NOT in the TODO summary; preserved here.)*
  - Shared `n_hat`-default / `L_esc` escape helpers between `mc_spectrum` and
    `mc_brem_spectrum` (`spectrum.py:144-148` & `267-273` vs `487-491` & `513`). *(TODO — NOT
    in the TODO summary; `montecarlo/` → ledger obligations apply, verbatim moves only.)*

### LOW — nits  *(DONE — `e6edc50`)*

- Dead expr `self.E[1] - self.E[0]` @ `eaglexo_response.py:315` (was the pyright warning).
- Dead statement `float(Eb[0])` @ `plots/spectra.py:155`.
- `_domega_of` @ `plots/detectors.py:266` — defined + re-exported + frozen
  (`test_plots_exports.py:50`) but never called; dead-but-export-locked, removing needs a
  deliberate freeze-test update.
- Redundant local `import pandas as pd` inside `results.py:292` `results_dataframe`.
- `run_sweep` (`run.py:139`) inlines the join that `checkpoint_path_for` exists for.
- `beta_from_keV` (transport, keV) vs `beta_from_Ee` (crystallography, eV) — same physics
  twice; layering permits transport delegating to crystallography.

---

## Large-file refactor roadmap

- **`plots/detectors.py` (677 → 599 after M1+M3) — best target.** M4 removes more; what's
  left splits cleanly at the existing `# ---- Timepix` / `# ---- Eagle XO` comment seams.
- **`results.py` (656)** — split into a `results/` package (store/selection, line-metrics,
  scoring, tables) behind a **new** export-freeze test — same pattern `montecarlo/` and
  `plots/` already used when they split.
- **`montecarlo/spectrum.py` (564) — handle with care.** Pure moves only (E_tab union-grid
  builder, the M7 escape/`n_hat` helpers). Don't restructure the `_accumulate` closure.
  Validation obligations apply.
- **`sweep.py` (550 → 520 after M5).**
- **Leave alone:** `plots/trajectories.py` (530, cohesive) and `checks/feranchuk_spence.py`
  (1056, standalone validation anchor) — not part of this cleanup.
- **`analysis.py` (453, matplotlib) vs `notebooks/analysis_app.py` (355, altair)** — parallel
  marimo apps; out of scope, but set a retirement date for the matplotlib one once altair
  parity is confirmed.

## Key learnings from the audit

- The plots package carries **migration-era duplication**: matplotlib drawers and their altair
  counterparts duplicate *reduction/physics* logic, not just rendering. Two impls of
  "best record per cell, flux-gated" will drift the same way H1 did.
- The **right abstractions already existed but weren't shared**: `altair_detectors._collapsed`,
  `altair_detectors._title`, `altair_sweeps._metrics_map`. The matplotlib side reinvented each
  inline (10×, 12×, 5×).
- `config._MATERIAL_GRIDS` (TypedDict registry) is the in-repo model for how
  `sweep.crystal_params`'s if-chain should be shaped; composition is derivable from the crystal
  `basis` (as `substrate_composition` already does with `Counter`).
- Grep-quantified duplication counts: collapse idiom ×10 (6 files), `cases→names` prologue ×9,
  title recipe ×12 (7 files), metrics-map ×5.
