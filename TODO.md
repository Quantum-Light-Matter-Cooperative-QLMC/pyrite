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

**NOTE:** User wrote item straight into `TODO.md` (denoted >user<)? Fold into branch (steps 1-2), then slim to one-line summary on `main` once branch exist.

## P1 - top-priority/high-value

### Active

1. **Physics validation ledger.** Continue fresh-context re-derivations, add missing in-code `Validation: <id>` markers. Design: [`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md); method: [`docs/validation/README.md`](docs/validation/README.md).
2. **Grazing grating — ALEX-s constants + hardware survey.** Research cited device constants and ~10 eV–4 keV CCD/grating landscape. → `docs/soft-xray-hardware-survey`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **Checkpoint rework** split up material pickles into individual line and brem pickles, with each mat having a subdir in ./checkpoints/
4. **>user< Multiple sweep types and related changes.** Different setups for different purposes, with configs for each independently adjustable:
   1. Full sweep uses current default settings -- fine steps, wide ranges, large Ne's, many angles/thicknesses swept over, lower threhold for included lattice vectors, etc
   2. Survey sweep (placeholder/provisional name, give some options) -- wider step, narrower ranges, smaller Ne's, etc.
      1. Survey sweeps have a high likelihood to proliferate many checkpoints or datasets with slightly varied parameter sets. This seemingly won't be well tolerated by the current system of checkpoint management. Evaluate alternative methods, provide options
   3. These should generally support `reline`/`rebrem`. We should maybe rename/slightly rework `reline` and `rebrem` commands (split to separate TODO line item?) to support more general use for when users just want to see line or brem data alone without calculating the other
   4) add flags to `rebrem` and `reline` that let you set new default values for each step, start, and/or stop, for `--all` or for the full set of provided `<mats>` (all mats provided would get same new defaults)
   5) Confirm that if user types `cxr pull --brem-only <mat1> <mat2>`, it will pull brem-only for BOTH materials (or full set of materials). Same with `--line-only` and related cmd's (options should generally apply to all provided mats across all repo cmds unless explicitly denoted otherwise)
5. **>user< Add NIST brem background dataset comparison/subtraction, as is done in Zhai et. al.**
6. **>user< Create citation ledger/document to cite all major literature used in development**. Will require agents to evaluate literature sources. Should be tied deeply to validation ledger.
7.  **>user< Confirm no more placeholder debye-waller factors.** Ensure all are literature-supported. Also, confirm no crystals have significant anisotropy in Debye-Waller (currently only single scalar value supported for each mat)
8. **>user< Fix remote (GPU?) OOM bug on `rebrem`.** Seen multiple times when running `cxr remote rebrem --all --ne-brem 500 --step 20`. Failed on multiple different materials multiple times (check remote logs, should have notes). Rerunning those mats generally led to success

### Gated

1. **Measured-data validation.** General experimental-simulation comparison & validation. Particularly: compare modeled broadened line widths vs measured HOPG rocking-curve / EDS dataset. Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).

### On-Hold

1. **High-energy electron/channeling support.** Evaluate `Geant4` or similar for REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter), JungFrau detector ~0.5–4.5 m from interaction point. USER QUESTION: What is rep rate?
2. **Bent Crystals (After add channeling + relativistic electrons)**
3. **Superradiant PXR/CBS** need bunch length knowledge, coherent emission *across segments* (also needed by channeling radiation as in long-term features #1, #3)

## P2 - medium-priority

1. **Add inv. lattice vector arrow(s) of interest to 3D Crystal Visualizer**
2. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **pyelsepa / ELSEPA transport.** Maintain landed, validated adapter externally provisioned CI environment. Design: [`docs/cstool-nebula-evaluation.md`](docs/cstool-nebula-evaluation.md).
4. **Material filters.** Model calibration-filter transmission between x-ray beam and detector. → `feature/material-filters`.

## P3 - lower / exploratory

1. **>user< Git history cleanup.** Evaluate squashing commits in history (currently still a 1-user repo, soon to be more).
