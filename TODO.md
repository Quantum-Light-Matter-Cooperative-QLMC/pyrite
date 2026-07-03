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

Full original inventory + rationale (tracked, self-contained — includes M4 and the M7
`line_fwhm_eV`/escape-helper sub-items not summarized below):
[`docs/dedup-inventory.md`](docs/dedup-inventory.md).

**Hard constraints on every item below:**
- `tests/test_plots_exports.py` + `tests/test_montecarlo_exports.py` freeze every
  re-exported name (`set(p.__all__) == FROZEN_EXPORTS`). Adding names is fine; removing
  or failing to re-export breaks the test. `altair_*` modules are intentionally NOT
  re-exported — don't add them.
- Anything touching `montecarlo/` or `crystallography.py` triggers the physics-validation-ledger
  workflow (derivation docstring + `Validation:` marker + ledger row + fresh-context
  verification per `docs/validation/README.md`). Prefer verbatim moves over rewrites there.
- Verify every step with `uv run pytest`, `uv run ruff check .`, `uv run pyright` — all
  three must stay clean (198 tests / 0 ruff / 0 pyright as of branch start).

### Remaining work, in suggested order

1. **M2 — shared `_si_sensor.py` detector-response module.** `timepix_response.py` and
   `eaglexo_response.py` duplicate: the Si constants (`W_EHP_EV`, `FANO_SI`,
   `SI_DENSITY_G_CM3`, `SI_A`, `SI_N_PER_ANG3`), the `_RESPONSE_CACHE` + `get_response`
   grid-key caching pattern, the `poisson_counts` core, and the `apply()` shape-guard
   text. Extract a new `_si_sensor.py` holding the constants + `grid_key(E)` +
   Poisson core; each response model keeps its own physics on top. This touches
   detector-response code, not core `montecarlo/` transport — confirm with
   `docs/physics-validation-ledger.md` whether the extracted pieces need ledger entries
   (pure refactor of already-validated pieces should not, but check before assuming).

2. **M3 — `plot_eaglexo_charge_map` → `plot_heatmaps` delegation.**
   `plots/detectors.py:562-677` reimplements ~90 lines of `sweeps.py`'s best-per-cell /
   `_cell_edges` / shared vmin-vmax / colorbar machinery, plus the thin-axis→line
   fallback. Give `plot_heatmaps` a `value=callable(record)` parameter; rewrite
   `plot_eaglexo_charge_map` as a ~20-line wrapper around it.

3. **M5 — `sweep.crystal_params` data registry.** The 110-line if-chain in `sweep.py`
   should become a registry mirroring `config._MATERIAL_GRIDS` (a `TypedDict` table),
   with composition derived from the crystal `basis` the way `substrate_composition`
   already does via `Counter` — don't hand-list compositions per material. Also unify
   the three radiator-dict constructions (`substrate_radiator`, `layer_radiator`, the
   inline film dict at `sweep.py:472-479`) into one constructor.

4. **`results.py` package split.** 656 lines; split into a `results/` package (store/
   selection, line-metrics, scoring, tables) behind a **new** export-freeze test —
   follow the same pattern `montecarlo/` and `plots/` already used when they split.

5. **M6 — renderer-neutral frame builders (largest item, aligns with the altair
   migration).** `plots/sweeps.py`'s `heatmap_frame`/`metric_vs_frame`/`scan_charts`/
   `_effective_x` re-derive the same reduction logic as `plot_heatmaps`/`plot_metric_vs`/
   `plot_scan`, just for altair instead of matplotlib — duplicating *reduction*, not just
   *rendering*, which is exactly what these modules' docstrings promise not to do. Move
   the frame builders into a renderer-neutral `plots/_frames.py` (or fold into
   `results.py`) consumed by both the matplotlib and altair renderers. Do this last —
   it's the biggest surface and easiest to get wrong once items 1-5 have already moved
   things around underneath it.

### Notes carried from the analysis session
- `plots/detectors.py` (677 lines) is the best large-file refactor target: M1 (done) +
  M3 + M2's constant-hoisting together remove ~200 lines; what's left splits cleanly at
  the existing `# ---- Timepix` / `# ---- Eagle XO` comment seams.
- Leave `plots/trajectories.py` (530, cohesive) and `checks/feranchuk_spence.py`
  (1056, standalone validation anchor) alone — not part of this cleanup.
- `analysis.py` (matplotlib, 453) vs `notebooks/analysis_app.py` (altair, 355) are
  parallel marimo apps; out of scope here, but worth a retirement date for the
  matplotlib one once altair parity is confirmed.
