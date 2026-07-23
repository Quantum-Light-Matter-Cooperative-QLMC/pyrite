# TODO / Backlog

Items live on `feature/...` / `bugfix/...` / `docs/...` branches, not `main`, till done.
In-progress detail live on branch (or design doc);
`main` keep one-line summary + pointer, enforced by /docs:todo-sync.
Priority weigh value-to-goal (line-flux / enhancement predictions + publication validation story) vs effort and risk.

Item generation:
----------------

1. Create branch of relevant type, switch to it
2. Overwrite branch TODO.md: 2-3 sentence problem summary + implementation path, scoped to item only. Publish to `origin`
3. Switch to `main`, add 1-sentence item summary, triage into existing TODO.md items, push tightly scoped `docs(todo)` commit to main

**NOTE:** User wrote detailed item straight into `TODO.md`? Fold into branch (steps 1-2), then slim to one-line summary on `main` once branch exist.

## P1 - high value (physics accuracy + publication validation)

### Active

1. **Physics validation ledger.** Continue fresh-context re-derivations, add missing in-code `Validation: <id>` markers. Design: [`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md); method: [`docs/validation/README.md`](docs/validation/README.md).
2. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **Grazing grating — ALEX-s constants + hardware survey.** Research cited device constants and ~10 eV–4 keV CCD/grating landscape. → `docs/soft-xray-hardware-survey`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).

### Gated

1. **Crystal mosaicity — measured-data validation.** Compare modeled broadened line widths vs measured HOPG rocking-curve / EDS dataset. Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).
2. **Multilayer film-on-substrate — measured-data validation.** Validate model vs measured film-on-substrate dataset. Design: [`docs/multilayer-materials.md`](docs/multilayer-materials.md).

## P2 - medium (experiment match + usability)

1. **pyelsepa / ELSEPA transport.** Maintain landed, validated adapter externally provisioned CI environment. Design: [`docs/cstool-nebula-evaluation.md`](docs/cstool-nebula-evaluation.md).
2. **Sweep cache standardization.** Include energy grid in cache reuse, report stale/mixed checkpoint records, no default-behavior change. → `feature/sweep-cache-standardization`.
3. **Material filters.** Model calibration-filter transmission between x-ray beam and detector. → `feature/material-filters`.
4. **Crystal definitions.** Review/confirm crystal lattice definitions + parameter values
5. **Grating Transport Bugfix** it can be seen in 2D and 3D visualizations that the electrons on blazed crystals do *start* properly at the blazed facets, but the *exit trajectories* of the electrons are calculated to the original crystal surface. This must be fixed, and it should be confirmed whether or not the x-ray escape path is being calculated correctly

## P3 - lower / exploratory

1. **Git history cleanup.** Evaluate structure/history improvements, squash minor upkeep or documentation commits where fitting.
2. **Dynamic GPU chunk sizing.** Evaluate config-driven chunk sizing for `cxr remote` GPU runs, document how config drive it.

## Long term features

1. **High-energy electron support.** Evaluate `Geant4` or similar for REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter), JungFrau detector ~0.5–4.5 m from interaction point. USER QUESTION: What is rep rate?
2. **Bent Crystals (After add channeling + relativistic electrons)**
3. **Crystal structure 3D visualizer (with inv. lattice vector arrows?)**
4. **Superradiant PXR/CBS** need bunch length knowledge, coherent emission *across segments* (also needed by channeling radiation as in long-term features #1, #3)
5. **Blazed-groove exit path optimization** blazed grooves to improve exit paths and yield in a given crystal.
