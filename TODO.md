# TODO — `perf/notebook-render-eval`

**Task:** evaluate whether splitting `notebooks/analysis_app.py` (1070 lines, 8-tab
marimo+Altair dashboard) into multiple sub-notebook files, plus downsampling /
"marimo_csv"-style caching, would speed up plot rendering — and plan it if worthwhile.

**Findings (measured on `checkpoints/mose2.pkl`, the densest checkpoint: 3720 records,
40 tilts x 31 azimuths x 3 energies):**
- Checkpoint load is fast (0.36 s) — not the bottleneck.
- `spectrum_chart` (one tilt, one band): ~0.15 s build+serialize, ~650 KB payload — fine.
- `heatmap_select_chart`: **0.94 s build**, only 200 KB payload — compute-bound, not data-volume.
- `scan_charts` (8 charts): **1.11 s build**, 2.9 MB total payload — also compute-bound.
- Root cause of both slow ones: `results.metrics.line_metrics` runs per-record scipy
  peak-finding (`find_peaks`/prominence/widths) for every record; `plots._common._metrics_map`
  memoizes it *within* one call (dict keyed by `id(record)`) but there is **no cache across
  calls** — every re-render redoes the full O(N) peak-finding pass from scratch.
- The whole tabbed dashboard is built inside **one ~600-line `@app.cell`**
  (`notebooks/analysis_app.py:460-1053`) closing over ~40 top-level widgets spanning all 8
  tabs — any single widget's change reruns the entire cell body. Whether this actually
  forces the *currently open* tab to recompute (vs. marimo preserving already-loaded lazy
  content across the rerun) could not be confirmed live — the Chrome browser-automation
  extension was not connected in this session, so the decisive interaction test (open a
  heavy tab, change an unrelated widget on another tab, switch back, check for a reload)
  is still open. Marimo's `IDProvider` resets its counter each cell execution, so the
  mega-cell's UI elements get *stable* ids across reruns — this weakens, but doesn't
  disprove, the "wasted remount" theory.

**Verdict: splitting into separate notebook FILES is not the fix.** Marimo apps are
independent reactive graphs; 6 of 8 tabs share `res`/`res_view`/the thickness pin/the
heatmap-click-to-spectrum wiring, so file-splitting would duplicate checkpoint loading and
break that cross-widget wiring for little payoff (only "Penetration" and "Cross-material"
are genuinely standalone). The measured bottleneck is uncached per-record peak-finding, not
file structure or (at current checkpoint sizes) data volume — Vega-Lite/vegafusion already
handles the 200 KB-2.9 MB payloads seen here.

**Recommended implementation path (ranked by confidence):**
1. **DONE — cache `line_metrics` results across calls.** `mo.cache`/`mo.lru_cache` turned
   out to be unusable outside a running `@app.cell` (raises `OSError: could not get source
   code` when the decorated function lives in a library module, since it depends on
   marimo's AST/cell-graph machinery). Implemented instead as a plain module-level dict
   cache in `src/cxr_mc/plots/_common.py` (`_LINE_METRICS_CACHE`, mirroring the existing
   `_EFF_CACHE` pattern), keyed on `(case["name"], case["E0_keV"], settings.beam_current_na,
   rel_prominence, line_metric)` — content-based, not `id()`-based, since CPython reuses
   freed objects' addresses and an identity key would be unsafe for a process-lifetime
   cache. Verified on `checkpoints/mose2.pkl` (3720 records): `heatmap_select_chart`
   warm-call time dropped from 1.06s to 0.07s (~15x); `cache_len` after warm-up equals
   `n_records` as expected. Full test suite (`uv run pytest`), `ruff check`, `ruff format`,
   and `pyright` all pass on the changed files (`src/cxr_mc/plots/_common.py`,
   `tests/test_sweep_metrics_reuse.py` — the latter's `line_metrics_calls` fixture now
   clears `_LINE_METRICS_CACHE` first so cross-test cache hits don't undercount calls).
2. **DONE - decouple the mega-cell into one `@app.cell` per tab.** `notebooks/analysis_app.py`
   now returns one zero-arg builder closure per tab, each from its own cell with only the
   widgets/data that tab reads, plus a final small cell that calls
   `mo.ui.tabs({...}, lazy=True)` over the 8 tab builders. The tab bodies and chart arguments
   were otherwise left unchanged. Verified with syntax compilation, direct Ruff and Marimo
   checks on `notebooks/analysis_app.py`, and the focused `tests/test_export.py` /
   `tests/test_analyze.py` launcher/export tests.
3. **Downsampling raw spectra**: lower priority — current payloads are within
   Vega-Lite/vegafusion's already-provisioned-for range; only revisit if 1-2 don't fully
   fix perceived lag.
4. File-splitting: not recommended now; revisit only if the file's line count itself (not
   render speed) becomes the actual maintenance pain point, and only for the two standalone
   tabs (Penetration, Cross-material).

**Remaining measurement:** the live cross-tab-recompute test still needs a connected browser
session if we want to quantify item 2's runtime payoff. The structural split is landed, so that
test is now evidence-gathering rather than an implementation gate.

---

# TODO / Backlog (inherited from main, for reference only — do not edit here)

Items live on `feature/...` / `bugfix/...` / `docs/...` branches, not `main`, until finished.
Full detail for an in-progress item lives on its branch (or its design doc);
`main` keeps only a one-line summary + pointer, enforced by /docs:todo-sync.
Priorities weigh value-to-goal (line-flux / enhancement predictions + the publication's validation story)
against effort and risk.

Item generation:
----------------

1. Create and move to branch of relevant type
2. Overwrite branch TODO.md with concise, 2-3 sentence problem summary + implementation path, scoped only to the relevant item, then publish to `origin`
3. Move to `main`, create 1 sentence summary of new item, then triage into existing TODO.md items and push tightly scoped `docs(todo)` commit to main

**NOTE:** If the user has written a detailed item summary directly into `TODO.md` on
main, fold it into a branch (steps 1-2 above), then slim it back to a one-line summary
on `main` once the branch exists.

## P1 - high value (physics accuracy + publication validation)

### Active

1. **Physics validation ledger — 19 of 26 claims unverified, 0 signed off.** Re-derive each
   claim in fresh context, then add the 17 missing in-code `Validation: <id>` markers.
   Ledger: [`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md);
   method: [`docs/validation/README.md`](docs/validation/README.md).
2. **Grazing grating — groove efficiency.** `Grating.groove_efficiency` is a placeholder
   scalar, not a groove-profile model. -> `feature/grating-groove-efficiency`.
   Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **Grazing grating — ALEX-s constants + hardware survey.** The ALEX-s device constants in
   `grating.py` are `### FILL IN` placeholders pending a real datasheet; the ~10 eV-4 keV
   CCD/grating survey is unwritten. -> `docs/soft-xray-hardware-survey`.
   Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).

### Gated

1. **Crystal mosaicity - measured-data validation.** MC route implemented; validate
   broadened line widths vs. a measured HOPG rocking-curve / EDS dataset
   (data-dependent). Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).
2. **Multilayer film-on-substrate - measured-data validation.** Model implemented;
   validate vs. a measured film-on-substrate dataset (data-dependent). Design:
   [`docs/multilayer-materials.md`](docs/multilayer-materials.md).

## P2 - medium (experiment match + usability)

1. **Crystallography — `crystals`-package redundancy question.** `diffpy`/`dans-diffraction`
   adapters landed; whether `crystals` still adds unique value (symmetry/Wyckoff expansion) is
   unresolved, pending a ~1-hour spike. Review:
   [`docs/crystallography-adapters-review.md`](docs/crystallography-adapters-review.md).
2. **pyelsepa / ELSEPA transport.** Adapter landed + validated (C 2.19%, Si 4.42% max rel vs
   NIST); gated in CI because the image/venv live outside the repo. -> `feature/elsepa-port`.
3. **Sweep cache standardization.** Checkpoint records key on `(name, E0)` only, so the energy
   grid is not part of the cache key and stores silently mix grid resolutions.
   -> `feature/sweep-cache-standardization`.
4. **Material filters.** Model calibration filters (e.g. sheets of Al foil) between the
   x-ray beam and detector, for detector calibration against filtered spectra.
   -> `feature/material-filters`.
5. **Finite electron beam size.** Confirm the input beam is finite, then model it as a
   ~1 mm diameter Gaussian beam incident on the crystal.
6. **De-duplication follow-through — M4 + M7.** The two clusters parked out of scope by the
   merged `refactor/dedup-followthrough`. -> `refactor/dedup-m4-m7`.
   Inventory: [`docs/dedup-inventory.md`](docs/dedup-inventory.md).

## P3 - lower / exploratory

1. **Git history cleanup.** Squash minor upkeep/doc commits; evaluate other repo
   structure/history improvements.
2. **Dynamic GPU chunk sizing.** Evaluate config-driven chunk-size selection for
   `cxr remote` GPU runs (probe a few test cases against the config's array sizes),
   including a write-up of what chunking is and how config values drive it.

## Long term features

1. `Geant4` or similar integration to support high-energy electron beams

   * Specifically, RAGAE@DESY
     * Energy 3-5 MeV
     * 50 fs duration
     * 100 fC charge
     * 200-300 um diameter on target

   JungFrau Detector is about 4.5 m away from IP but could be as short as ~50 cm (in vacuum)
