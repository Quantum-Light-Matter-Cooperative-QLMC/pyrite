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
2. **Polars investigation.** Evaluate Polars for packaging large parameter-sweep metadata.
3. **pyelsepa / ELSEPA transport.** -> `feature/elsepa-port` Adapter landed + **validated** (C 2.19%,
   Si 4.42% max rel vs NIST); image now builds tarball-free from
   `github.com/eScatter/elsepa`. Remaining gate: the image/venv live outside the repo
   (`C:/dev/pyelsepa`), so the driver stays gated in CI. Tied to P2 #2.
4. **Codebase de-duplication follow-through.** `refactor/dedup-followthrough` — merged
   (now on `main`). Two items intentionally out of scope, tracked only in
   [`docs/dedup-inventory.md`](docs/dedup-inventory.md): M4 (wide-brem overlay
   physics x4) and the M7 `line_fwhm_eV`/escape-helper sub-items.
5. **Material filters.** Model calibration filters (e.g. sheets of Al foil) between the
   x-ray beam and detector, for detector calibration against filtered spectra.
6. **Finite electron beam size.** Confirm the input beam is finite, then model it as a
   ~1 mm diameter Gaussian beam incident on the crystal.
7. **Sweep cache standardization.** Round parametric angular sweeps to the nearest
   degree; standardize energy-grid sizes/spacings so thickness/angle/etc. sweeps share
   one cached-data store that is always checked before running.

## P3 - lower / exploratory

1. **Marimo/Altair follow-ups.** Core migration landed from `feature/marimo-transfer`;
   remaining notebook/plot fixes:
   - Enforce fixed plot size for heatmaps (they expand with axis sizes).
   - Enforce few-decimal-point axis tick labels (heatmaps worst offenders).
   - Angle selector for penetration plots (trajectories + mean population vs depth);
     default to a low nonzero polar angle (~15 deg).
   - Intrinsic-spectra plot of CXR emission without the incoherent brem background;
     user-adjustable x-axis limits on both spectral plots; lin/log y switch.
   - Fix blank detector-tab plots.
   - Fix penetration plots for multilayer/stacked materials: show trajectories and
     population-vs-depth through the full stack, not just the top layers (mos2).
   - Finalize transition to `scan_app.py` (can't select a material, among other issues).
2. **Grazing-incidence soft X-ray diffraction grating.** -> `feature/grazing-grating`.
   Dispersion scaffold implemented; next is grating reflectivity + detected-image model.
3. **CLI/remote output noise.** Silence the import-time "No GPU found, or cupy not
   installed!" banner on every `cxr` invocation and the repeated "no Mott transport
   table for 'X'" warnings spammed by `dev/remote.py` runs.
4. **`remote.py start --follow` hang.** Launches the task but hangs afterward and never
   attaches the tqdm progress bar.
5. **Git history cleanup.** Squash minor upkeep/doc commits; evaluate other repo
   structure/history improvements.

## Long term features

1. `Geant4` or similar integration to support high-energy electron beams
2. Add support for `Numba with CUDA` to significantly speed up calcs
