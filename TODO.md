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
   (now on `main`). All mechanical hoists landed: title/metrics hoists into
   `plots/_common.py`, shared `_si_sensor.py`, `plot_heatmaps` value= mode + Eagle
   charge-map delegation, `sweep.crystal_params` registry collapse, the `results.py` ->
   `results/` package split, and renderer-neutral plot frame builders
   (`plots/_frames.py`) shared by matplotlib/Altair. Two items intentionally out of
   scope, tracked only in [`docs/dedup-inventory.md`](docs/dedup-inventory.md): M4
   (wide-brem overlay physics x4) and the M7 `line_fwhm_eV`/escape-helper sub-items.

## P3 - lower / exploratory

1. **Marimo/Altair follow-ups.** Core migration landed from `feature/marimo-transfer`;
   `marimo` notebook/plot cleanup & fixes remain.
2. **Grazing-incidence soft X-ray diffraction grating.** -> `feature/grazing-grating`.
   Dispersion scaffold implemented; next is grating reflectivity + detected-image model.

## Long term features

1. Geant4 or similar integration to support high-energy electron beams
