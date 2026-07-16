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

**NOTE:** If the user has written a detailed item summary directly into `TODO.md`,
fold it into a branch (steps 1-2 above), then slim it back to a one-line summary
on `main` once the branch exists.

## P1 - high value (physics accuracy + publication validation)

### Active

1. **Physics validation ledger.** Continue fresh-context re-derivations and add missing in-code `Validation: <id>` markers. Design: [`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md); method: [`docs/validation/README.md`](docs/validation/README.md).
2. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency`'s placeholder scalar with a groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **Grazing grating — ALEX-s constants + hardware survey.** Research cited device constants and the ~10 eV–4 keV CCD/grating landscape. → `docs/soft-xray-hardware-survey`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).

### Gated

1. **Crystal mosaicity — measured-data validation.** Compare modeled broadened line widths with a measured HOPG rocking-curve / EDS dataset. Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).
2. **Multilayer film-on-substrate — measured-data validation.** Validate the implemented model against a measured film-on-substrate dataset. Design: [`docs/multilayer-materials.md`](docs/multilayer-materials.md).

## P2 - medium (experiment match + usability)

1. **pyelsepa / ELSEPA transport.** Maintain the landed, validated adapter's externally provisioned CI environment. Design: [`docs/cstool-nebula-evaluation.md`](docs/cstool-nebula-evaluation.md).
2. **Sweep cache standardization.** Include the energy grid in cache reuse and report stale/mixed checkpoint records without changing default behavior. → `feature/sweep-cache-standardization`.
3. **Material filters.** Model calibration-filter transmission between the x-ray beam and detector. → `feature/material-filters`.
4. **Crystal definitions.** Review/confirm various crystal lattice definitions & parameter values

## P3 - lower / exploratory

1. **Git history cleanup.** Evaluate structure/history improvements and squash minor upkeep or documentation commits where appropriate.
2. **Dynamic GPU chunk sizing.** Evaluate config-driven chunk sizing for `cxr remote` GPU runs and document how configuration drives it.

## Long term features

1. **High-energy electron support.** Evaluate `Geant4` or similar integration for REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter) and a JungFrau detector roughly 0.5–4.5 m from the interaction point.

2. **Complex shapes** 3D patterned sufaces, maybe diffraction-grating style, and other interesting shapes
