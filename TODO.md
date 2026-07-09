# TODO / Backlog

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

1. **Grazing-incidence soft X-ray diffraction grating - CCD + geometry buildout.**
   -> `feature/grazing-grating`. Promoted from P3: dispersion geometry landed; next
   is grating reflectivity/groove efficiency, then a real detected-image forward
   model (start with a simple CCD that bins photons by per-pixel polar angle,
   then a more physical CCD model), sized to real hardware — McPherson 251MX
   gratings (120/300/1200/2400 g/mm, ~6 eV-1.24 keV) and greateyes ALEX-s
   1k256/2k512 CCDs — before surveying broader ~10 eV-4 keV CCD/grating options.
   Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).

### Gated

1. **Crystal mosaicity - measured-data validation.** MC route implemented; validate
   broadened line widths vs. a measured HOPG rocking-curve / EDS dataset
   (data-dependent). Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).
2. **Multilayer film-on-substrate - measured-data validation.** Model implemented;
   validate vs. a measured film-on-substrate dataset (data-dependent). Design:
   [`docs/multilayer-materials.md`](docs/multilayer-materials.md).

## P2 - medium (experiment match + usability)

1. **External crystallography library adapters.** `codex/diffpy-structure-importer`
   implements `diffpy.structure` CIF/import support; `codex/dans-diffraction-research`
   implements optional `Dans_Diffraction` validation-oracle checks. Next: review/merge
   those branches, then decide whether the original `crystals` package still offers
   unique value.
2. **pyelsepa / ELSEPA transport.** -> `feature/elsepa-port` Adapter landed + **validated** (C 2.19%,
   Si 4.42% max rel vs NIST); image now builds tarball-free from
   `github.com/eScatter/elsepa`. Remaining gate: the image/venv live outside the repo
   (`C:/dev/pyelsepa`), so the driver stays gated in CI.
3. **Codebase de-duplication follow-through.** `refactor/dedup-followthrough` — merged
   (now on `main`). Two items intentionally out of scope, tracked only in
   [`docs/dedup-inventory.md`](docs/dedup-inventory.md): M4 (wide-brem overlay
   physics x4) and the M7 `line_fwhm_eV`/escape-helper sub-items.
4. **Material filters.** Model calibration filters (e.g. sheets of Al foil) between the
   x-ray beam and detector, for detector calibration against filtered spectra.
5. **Finite electron beam size.** Confirm the input beam is finite, then model it as a
   ~1 mm diameter Gaussian beam incident on the crystal.
6. **Sweep cache standardization.** Round parametric angular sweeps to the nearest
   degree; standardize energy-grid sizes/spacings so thickness/angle/etc. sweeps share
   one cached-data store that is always checked before running.
7. **`analysis_app.py` parameter-sweep views.** Support parameter sweeps (e.g. the
   crystal-thickness sweeps in the current h-BN work); today the app silently shows only
   the thinnest crystal.
8. **Checkpoint union tooling.** Union a live material checkpoint with an archived one
   for the same material; archive the current pickle first by default, leave the source
   archive intact — both overridable (skip archive / delete old archive).

## P3 - lower / exploratory

1. **Marimo/Altair follow-ups.** Core migration landed from `feature/marimo-transfer`;
   `feature/marimo-altair-followups` closed out the remaining fixes (blank detector
   tabs, mos2 multilayer penetration, angle selector, intrinsic-spectra controls,
   heatmap sizing/ticks, `scan_app.py` material dropdown). `cxr export` now renders
   `notebooks/analysis_app.py` via `marimo export html` (replacing the retired
   nbconvert-PDF path), and sweep-chart drivers share one precomputed metrics map
   across quantities instead of recomputing per-quantity. `eaglexo_charge_chart`
   now takes `x_domain=` like the other detector charts, and the default first
   tab is "Intrinsic spectra" instead of "Top geometries". Remaining deferred
   items:

   - The dense penetration-grid accordion (matplotlib, Penetration tab) is still
     fixed at `energy=30` regardless of the angle selector above it.
   - The x/y plot-limit entry boxes should move into the Spectra tab they belong to.
   - Add capability to click on individual heatmap pixels to select that parameter set for spectral plotting
2. **Git history cleanup.** Squash minor upkeep/doc commits; evaluate other repo
   structure/history improvements.
3. **Dynamic GPU chunk sizing.** Evaluate config-driven chunk-size selection for
   `dev/remote.py` GPU runs (probe a few test cases against the config's array sizes),
   including a write-up of what chunking is and how config values drive it.

## Long term features

1. `Geant4` or similar integration to support high-energy electron beams

   * Specifically, RAGAE@DESY
     * Energy 3-5 MeV
     * 50 fs duration
     * 100 fC charge
     * 200-300 um diameter on target

   JungFrau Detector is about 4.5 m away from IP but could be as short as ~50 cm (in vacuum)
