# Material config rework: exposing crystal orientation & dominant-plane count

Should the per-material config be reworked, and how should crystal plane
orientation (`beam_uvw`) and the dominant-reflection count (`n_families`) be
exposed instead of living as source-level defaults? **Recommendation: split
into a cheap, physics-risk-free plumbing pass (CLI flags + surfacing the
orientation that's already persisted per-checkpoint but never printed) now,
and a moderate-churn registry unification (still no physics change) as a
follow-up.** Do not change any of `n_families`'s default,
`g_max_invang`, or the reflection-ranking metric as part of this — that's a
physics change, not a config-exposure change, and would require a fresh
validation pass.

## What the codebase does today

Per-material configuration is split across three places by import layer
(leaf → driver, see `docs/repo_map.md`):

```
crystal_structures.toml   physics data: lattice, basis, mosaic_fwhm_deg
        │                 (crystallography.py loads it; has its own
        │                  Validation: <id> entries in the ledger)
        ▼
sweep.py  _CRYSTAL_PARAMS  crystal orientation: beam_uvw (zone axis),
                            hkl_list (optional hand-pinned override),
                            B_ang2, per-material E_grid default
        │
        ▼
config.py  _MATERIAL_GRIDS  scan geometry/energy grids: thickness,
                            tilt/azimuth sweeps, energy_keV, E_grid_line/brem
```

`config.py` already does `from .sweep import Layer, ScalarOrSeq, Sweep`, so
`_CRYSTAL_PARAMS` cannot simply move into `config.py` — `sweep.crystal_params`
would then need to import back from `config.py`, a cycle. The split is a
structural consequence of the DAG, not an oversight.

### Orientation is *partially* exposed already

`Sweep.beam_uvw` (`None` → per-material default) and `Sweep.n_families`
(dataclass default `4`, applied uniformly) are real fields, and
`material_sweep()`'s `**overrides` forwards straight into
`dataclasses.replace`. So this already works, today, with no code change:

```python
material_sweep("mose2", beam_uvw=(1, 0, 0), n_families=6)
```

A per-stack `Layer` also carries its own `beam_uvw` / `azimuth_deg`
(`sweep.py:107-124`), so a film-on-substrate run can already orient each layer
independently.

### Two real gaps

1. **`hkl_list` has no override path at all.** For HOPG and h-BN it is
   hand-pinned in `_CRYSTAL_PARAMS` (`pm((0, 0, 2), (0, 0, 4))`), bypassing
   `dominant_reflections` entirely — which means `n_families` is a **silent
   no-op** for exactly those two materials (`sweep.py:266-273`). Every other
   material auto-selects via `dominant_reflections`, with no way to pin a
   specific plane set without editing source. "Which planes" and "how many
   planes" currently live in two different mechanisms (a hardcoded bypass vs.
   a `Sweep` field) that don't compose.
2. **Persisted, but not surfaced.** Correction to an earlier pass of this
   evaluation: grepping `results.py`/`_checkpoint_io.py`/`run.py` for the
   literal names `hkl_list`/`beam_uvw`/`n_families` returns no hits, but that
   grep was the wrong test — `build_cases` already puts the resolved
   `hkl_list`/`beam_uvw` into every case dict (`sweep.py:466-478`), and
   `store_result` stores the whole case verbatim as `record["case"]`
   (`results/store.py:80`). That survives the checkpoint round-trip: `run.py`
   reads `r["case"]["crystal"]` off *loaded* checkpoints, and
   `results/selection.py`'s `slim_results` explicitly keeps `"case"` as one of
   its retained keys. So the actual planes summed and the zone axis used
   **are** recoverable from any checkpoint today — the real gap is that
   nothing surfaces them without manually indexing into
   `record["case"]["hkl_list"]`: `cxr scan` prints no orientation summary, and
   `sweep.geometry_table` shows only a reflection *count* (`len(c["hkl_list"])`),
   not which planes or the zone axis. `n_families` itself (the request, as
   opposed to the `hkl_list` it resolved to) is not stored as its own field —
   but the resolved `hkl_list` is the more useful artifact of the two, and for
   HOPG/h-BN, echoing `n_families` would be misleading anyway (next point).
   This is the literal "done silently" the config is being reworked to fix,
   and it's a *surfacing* problem, not a persistence problem.

## Options for where a unified per-material config lives

1. **Leave the split, add flags + surfacing only.** Zero churn to the
   registries themselves. Fixes the CLI-visibility and surfacing gaps but
   leaves "editing one material means touching two files" unresolved.
2. **New leaf module (`materials.py`) between `crystallography.py` and
   `sweep.py`/`config.py`.** Both `sweep.py` and `config.py` import from it;
   no cycle. Holds one row per material with orientation fields (`beam_uvw`,
   optional `hkl_list` pin + a required reason string when pinned, so the
   auto/pinned distinction is visible instead of implicit) sitting next to
   (or cross-referenced with) the existing scan-grid fields. This is the
   "one place per material" outcome implied by the request. Moderate churn:
   touches `config.py`, `sweep.py`, all ~13 material entries, and any test
   that imports `_MATERIAL_GRIDS`/`_CRYSTAL_PARAMS` directly
   (`tests/test_sweep.py`, `tests/test_crystallography.py`).
3. **Fold `crystal_structures.toml` in too.** Rejected: that file is
   physics data with its own ledger entries (e.g. `Validation:
   sapphire-corundum-structure`) reused independently by
   `validation_oracles.py`/`checks/dans_diffraction_oracle.py`. Merging it
   with run-scan knobs would drag physics data under scan-config churn and
   blur what the validation ledger is tracking.

**Recommendation: option 2**, done as a follow-up after the plumbing pass
below lands (smaller, reviewable diff first; the registry merge can reuse
whatever field shapes the CLI/persistence work settles on).

## What's cheap vs. what carries a validation cost

| Change | Risk |
| --- | --- |
| `--n-families` / `--beam-uvw` flags on `cxr scan`, forwarded to existing `Sweep` overrides | None — same code path, just argparse wiring |
| Print the resolved `hkl_list`/`beam_uvw` from `cxr scan` (already persisted in `record["case"]`; just not echoed anywhere) | None — reads existing data, computes nothing new |
| Unify `_MATERIAL_GRIDS` + `_CRYSTAL_PARAMS` into one per-material registry | None — same values, relocated |
| Changing the `n_families` **default** (currently 4 for every non-pinned material), `g_max_invang`, or the `\|S(g)\| e^{-W}/g^2` ranking metric | **Physics change** — alters the summed reflection set for every current material's spectra; needs a fresh-context re-derivation + anchor-figure re-check per `docs/physics-validation-ledger.md` before it could be adopted |

## Recommendation (staged)

1. **CLI flags** — add `--n-families INT` and `--beam-uvw H K L` to
   `scan.py:_build_parser`, threaded into the existing `material_sweep(...,
   n_families=..., beam_uvw=...)` override mechanism. No new plumbing needed
   beyond argparse.
2. **Surfacing** — the orientation is already persisted in `record["case"]`
   (see above); print the resolved `beam_uvw`/`hkl_list` at the end of
   `cxr scan` (reading `cases[0]`, not `sweep`, since HOPG/h-BN resolve
   `hkl_list` independent of any `n_families` request), and extend
   `geometry_table` to show more than a bare reflection count when someone
   wants to eyeball a checkpoint's orientation without writing a one-off
   script.
3. **Registry unification (`materials.py`)** — after 1-2 land and settle the
   field shapes, migrate `_MATERIAL_GRIDS`/`_CRYSTAL_PARAMS` into one
   per-material row each, including a visible reason string for materials
   that hand-pin `hkl_list` (HOPG, h-BN) so the auto-vs-pinned distinction
   stops being implicit.
4. **Not in scope here** — no change to `n_families`'s default, `g_max_invang`,
   or the ranking metric. If someone wants that later, it's a physics-review
   task, not a config-exposure task.
