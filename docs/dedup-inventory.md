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
| M2   | Si constants + `grid_key`/`prep_spectrum`/`poisson_core` hoisted into `_si_sensor.py` | this branch, uncommitted |
| M3   | `plot_eaglexo_charge_map` rewritten as a wrapper around `plot_heatmaps`' new `value=` mode | this branch, uncommitted |

**Remaining** (suggested order in `TODO.md`): M5, `results.py` split, M6. **Plus two
items that fell out of the TODO summary and must not be lost: M4, and the M7 `line_fwhm_eV`
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

- **M2 — timepix vs eaglexo response share verbatim blocks.**  *(DONE — this branch,
  uncommitted.)* `timepix_response.py` and `eaglexo_response.py` duplicated: the Si constants
  (`W_EHP_EV`, `FANO_SI`, `SI_DENSITY_G_CM3`, `SI_A`, `SI_N_PER_ANG3`), the
  `_RESPONSE_CACHE` + `get_response` grid-key caching pattern, the `poisson_counts` core,
  and the `apply()` shape-guard text. Extracted a new `_si_sensor.py` holding the constants +
  `grid_key(E)` + `prep_spectrum(spec, E_grid, module_name)` (the shape guard) +
  `poisson_core(E_grid, detected_per_s, time_s, rng)`; both modules import the constants and
  delegate to the three helpers, keeping their own physics (charge-sharing MC, QE table, ...)
  untouched. No ledger entries needed — verbatim relocation, no existing `Validation:` markers
  to preserve (both ledger rows were already `unverified`/`blocked` pre-refactor), confirmed
  against `docs/physics-validation-ledger.md` before assuming so.

- **M3 — `plot_eaglexo_charge_map` re-implements `plot_heatmaps`.**  *(DONE — this branch,
  uncommitted.)* `plots/detectors.py:562-677` duplicated ~90 lines of `sweeps.py` best-per-cell /
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

- **M5 — `sweep.crystal_params` 110-line if-chain.**  *(TODO)*
  Every branch returns `{crystal, composition, hkl_list, beam_uvw, B_ang2, E_grid}`. Collapse
  to a data registry mirroring `config._MATERIAL_GRIDS` (a `TypedDict` table), with composition
  derived from the crystal `basis` via `Counter` the way `substrate_composition` already does —
  don't hand-list compositions per material. Also unify the three radiator-dict constructions
  (`substrate_radiator`, `layer_radiator`, the inline film dict at `sweep.py:472-479`) into one
  constructor. Removes ~80 lines from the 550-line `sweep.py`.

- **M6 — matplotlib `sweeps` vs `altair_sweeps` duplicate the reduction, not just rendering.**
  *(TODO — do last.)*  `heatmap_frame` / `metric_vs_frame` / `scan_charts` / `_effective_x`
  re-derive `plot_heatmaps` / `plot_metric_vs` / `plot_scan` logic — duplicating *reduction*,
  which is exactly what these modules' docstrings promise not to do. (`metric_vs_chart` also
  double-computes records + `_effective_x`.) Move the frame builders into a renderer-neutral
  `plots/_frames.py` (or fold into `results.py`) consumed by both renderers. Largest surface,
  easiest to get wrong — do after items 1-5 have moved things underneath it.

- **M7 — smaller mechanical hoists.**
  - `records_for_cases(results, cases)` — the `cases→names` prologue ×9. *(DONE — `6f51ced`.)*
  - `_case_title` — extract from `altair_detectors._title(...)`; title recipe reimplemented
    inline ~12× across `spectra.py`, `detectors.py`, `sweeps.py`, `altair_sweeps.py`,
    `interactive.py`. Target: `plots/_common.py` next to `_best_azimuth`/`_peak_line`.
    *(DONE — this branch, uncommitted; 8 call sites swapped. `sweeps.py` had no case-title
    code, so only the other 4 files needed edits. Left alone: `interactive.py`'s plotly
    slider title and `spectra.py`'s mosaic title — genuine format divergences, not the same
    recipe.)*
  - `_metrics_map` — extract from `altair_sweeps._metrics_map(...)`; reimplemented ×5. Same
    target module. *(DONE — this branch, uncommitted; 5 call sites swapped across
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
- **`sweep.py` (550)** — M5 removes ~80 lines.
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
