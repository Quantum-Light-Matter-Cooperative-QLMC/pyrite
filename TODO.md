# TODO / Backlog

Items live on `feature/...` / `bugfix/...` / `docs/...` branches, not `main`, till done.
In-progress detail live on branch (or design doc);
`main` keep one-line summary + pointer, enforced by /todo-sync.
Priority weigh value-to-goal (line-flux / enhancement predictions + publication validation story) vs effort and risk.

Item generation:
----------------

1. Create branch of relevant type, switch to it
2. Overwrite branch TODO.md: 2-3 sentence problem summary + implementation path, scoped to item only. Publish to `origin`
3. Switch to `main`, add 1-sentence item summary, triage into existing TODO.md items, push tightly scoped `docs(todo)` commit to main

**NOTE:** User wrote item straight into `TODO.md` (denoted >user<)? Fold into branch (steps 1-2), then drop >user< & slim to one-line summary on `main` once branch exist.

## P1 - top-priority / high-value

### Active

1. **CLI rework: `energy-grid` / `cxr profile` / remote integration.** Disentangle fidelity (`full|survey` → `--fidelity`) from catalog profiles, invert schema (profiles reference materials), shared derived-grid store with provenance, `cxr profile` verb group, remote submit/pull by profile. → `feature/cli-profile-rework`; plan: [`docs/cli-energy-grid-sweep-rework-plan.md`](docs/cli-energy-grid-sweep-rework-plan.md).
2. **Physics validation ledger.** Continue fresh-context re-derivations, add missing in-code `Validation: <id>` markers. Design: [`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md); method: [`docs/validation/README.md`](docs/validation/README.md).
3.  **Debye-Waller provenance and anisotropy audit.** Continue replacing placeholder or reused `B_ang2` values with primary-source values and resolve per-site/tensor model needs. → `feature/debye-waller-audit`; audit: [`docs/debye-waller-audit.md`](docs/debye-waller-audit.md).
   1. >user< Evaluate complexity/value of implementing full anisotropic/tensor-based Debye-Waller factors when available
   2. >user< Evaluate worth in both implementing the approximate scalar Debye-Waller formula (compare output to known values for various anisotropic materials we have in our DB), and in attempting to implement a fully-fledged DFPT system.
   3. >user< Take a crystal with a known DW factor, then manually change it up and down over a range of values that can reasonably be expected other crystals to have, and see how much it changes by -- if large, then its worth being careful here.
4.  **GPU-memory follow-up.** Benchmark remote `rebrem --all --ne-brem 500 --step 20` for bounded CuPy reserved-pool memory; assess analogous `reline` cleanup separately.

### Gated

1. **Measured-data validation.** General experimental-simulation comparison & validation. Particularly: compare modeled broadened line widths vs measured HOPG rocking-curve / EDS dataset. Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).

### On-Hold

1. **High-energy electron/channeling support.** Evaluate `Geant4` or similar for REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter), JungFrau detector ~0.5–4.5 m from interaction point. USER QUESTION: What is rep rate?
2.  **Bent Crystals (After add channeling + relativistic electrons)**
3.  **Superradiant PXR/CBS** need bunch length knowledge, coherent emission *across segments* (also needed by channeling radiation as in long-term features #1, #3)

## P2 - medium-priority

1.  **>user< Add newly created compute-progress readout from ./notebooks/scan_app.py to the remote readout as well (using the same nice colored bars that are already used there)**
2. **Grazing grating — ALEX-s constants + hardware survey.** Research cited device constants and ~10 eV–4 keV CCD/grating landscape. → `docs/soft-xray-hardware-survey`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **Parameter-space sampling review.** Design principled prioritization across high-dimensional sweep parameters. → `docs/parameter-space-sampling-review`.
4. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
5. **pyelsepa / ELSEPA transport.** Maintain landed, validated adapter externally provisioned CI environment.
6.  **Material filters.** Model calibration-filter transmission between x-ray beam and detector. → `feature/material-filters`.
7.  **>user< Fix cached pull-in of pre-rendered animations in trace_app (add more buttons or something) -- maybe due to regenerate's changing random seed?**
   1.  Progress bar here shows up at the top of the marimo notebook instead of near the button that is pressed to start the render -- confusing
   2.  clip off the extra figure background & legend in the render, it is ugly. We just want the black grid space, with the colorbar and mat/config title info overlayed, but no background color. Saved render is also a bit pixelated, especially when opened in an mp4 viewer outside of the marimo app.
   3.  add button to open render saving dialogue (so user can promptly move it from the cache)

## P3 - lower / exploratory / small bugfixes