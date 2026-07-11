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

# USER ADDED:

- Zhai supplementary coherent-emission figures (WSe₂ 42/55/75 nm, MoSe₂
  47/112/147 nm, h-BN 921 nm, all 200 keV, at polar tilts -10/-15/-17.5/-20 deg)
  are implemented in `checks/anchor_figures.py` + the validation app. Still
  unconfirmed: whether these assume azimuthal angle == 0, per the paper.
  - I have added a 'cxr check' command which runs the validation notebook. Please implement some additional commands/options to use the lab box w/ GPU for the calculation of these various monte-carlos, since my laptop doesn't cut it very well
- Implement automated ACP server startups for interaction with marimo notebooks:
  ------------------------------------------------------------------------------


  - npx stdio-to-ws "cmd /c npx @zed-industries/claude-code-acp" --port 3017
  - npx stdio-to-ws "cmd /c npx @zed-industries/codex-acp" --port 3021

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
7. **Material config rework — expose crystal orientation + dominant-plane count.**
   `beam_uvw`/`n_families` are overridable via `Sweep` but not from the CLI, `hkl_list`
   has no override path, and none of it is persisted with a checkpoint (unreproducible
   after a registry default changes). Staged plan: CLI flags -> checkpoint persistence
   -> unify `config._MATERIAL_GRIDS`/`sweep._CRYSTAL_PARAMS` into one per-material
   registry (`materials.py`, no cycle). No physics change. ->
   `feature/material-config-rework`. Evaluation:
   [`docs/material-config-evaluation.md`](docs/material-config-evaluation.md).

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
