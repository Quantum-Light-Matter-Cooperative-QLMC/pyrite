# Material config rework: exposing crystal orientation & dominant-plane count

Should the per-material config be reworked, and how should crystal plane
orientation (`beam_uvw`) and the dominant-reflection count (`n_families`) be
exposed instead of living as source-level defaults? **Recommendation: split
into a cheap, physics-risk-free plumbing pass (CLI flags + checkpoint
persistence) now, and a moderate-churn registry unification (still no physics
change) as a follow-up.** Do not change any of `n_families`'s default,
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
2. **Nothing is recorded.** `cxr scan` (`scan.py`) has no `--n-families` /
   `--beam-uvw` flags — reaching either knob requires calling
   `material_sweep()` from Python. More importantly, grepping `results.py`,
   `_checkpoint_io.py`, and `run.py` for `hkl_list` / `beam_uvw` / `n_families`
   returns **zero matches**: none of the orientation actually used to produce
   a checkpoint is persisted with it. If the per-material defaults ever change,
   an old checkpoint's spectrum can no longer be traced back to which
   reflections generated it. This is the literal "done silently" the config
   is being reworked to fix, and it's independent of whether the knobs become
   CLI-settable.

## Options for where a unified per-material config lives

1. **Leave the split, add flags + persistence only.** Zero churn to the
   registries themselves. Fixes the CLI-visibility and reproducibility gaps
   but leaves "editing one material means touching two files" unresolved.
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
| Persist `hkl_list` / `beam_uvw` / `n_families` into checkpoint/run metadata | None — recording, not computing |
| Unify `_MATERIAL_GRIDS` + `_CRYSTAL_PARAMS` into one per-material registry | None — same values, relocated |
| Changing the `n_families` **default** (currently 4 for every non-pinned material), `g_max_invang`, or the `\|S(g)\| e^{-W}/g^2` ranking metric | **Physics change** — alters the summed reflection set for every current material's spectra; needs a fresh-context re-derivation + anchor-figure re-check per `docs/physics-validation-ledger.md` before it could be adopted |

## Recommendation (staged)

1. **CLI flags** — add `--n-families INT` and `--beam-uvw H K L` to
   `scan.py:_build_parser`, threaded into the existing `material_sweep(...,
   n_families=..., beam_uvw=...)` override mechanism. No new plumbing needed
   beyond argparse.
2. **Persistence** — stamp the `cp["hkl_list"]` / `cp["beam_uvw"]` /
   `sweep.n_families` actually used into whatever metadata already rides
   along with a checkpoint (check `results.Settings` / the run-case dict in
   `run.py`), so a saved checkpoint is traceable to the reflections that
   produced it.
3. **Registry unification (`materials.py`)** — after 1-2 land and settle the
   field shapes, migrate `_MATERIAL_GRIDS`/`_CRYSTAL_PARAMS` into one
   per-material row each, including a visible reason string for materials
   that hand-pin `hkl_list` (HOPG, h-BN) so the auto-vs-pinned distinction
   stops being implicit.
4. **Not in scope here** — no change to `n_families`'s default, `g_max_invang`,
   or the ranking metric. If someone wants that later, it's a physics-review
   task, not a config-exposure task.
