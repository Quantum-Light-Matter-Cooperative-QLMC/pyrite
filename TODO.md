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

1. **Physics validation ledger.** Continue fresh-context re-derivations, add missing in-code `Validation: <id>` markers. Design: [`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md); method: [`docs/validation/README.md`](docs/validation/README.md).
2. **Debye-Waller provenance and anisotropy audit.** Continue replacing placeholder or reused `B_ang2` values with primary-source values and resolve per-site/tensor model needs. → `feature/debye-waller-audit`; audit: [`docs/debye-waller-audit.md`](docs/debye-waller-audit.md).
   1. >user< Evaluate complexity/value of implementing full anisotropic/tensor-based Debye-Waller factors when available
   2. >user< Evaluate worth in both implementing the approximate scalar Debye-Waller formula (compare output to known values for various anisotropic materials we have in our DB), and in attempting to implement a fully-fledged DFPT system.
   3. >user< Take a crystal with a known DW factor, then manually change it up and down over a range of values that can reasonably be expected other crystals to have, and see how much it changes by -- if large, then its worth being careful here.
3. **GPU-memory follow-up.** Benchmark remote `rebrem --all --ne-brem 500 --step 20` for bounded CuPy reserved-pool memory; assess analogous `reline` cleanup separately.

### Gated

1. **Measured-data validation.** General experimental-simulation comparison & validation. Particularly: compare modeled broadened line widths vs measured HOPG rocking-curve / EDS dataset. Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).

### On-Hold

1. **High-energy electron/channeling support.** Evaluate `Geant4` or similar for REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter), JungFrau detector ~0.5–4.5 m from interaction point. USER QUESTION: What is rep rate?
2. **Bent Crystals (After add channeling + relativistic electrons)**
3. **Superradiant PXR/CBS** need bunch length knowledge, coherent emission *across segments* (also needed by channeling radiation as in long-term features #1, #3)

## P2 - medium-priority

1. **Grazing grating — ALEX-s constants + hardware survey.** Research cited device constants and ~10 eV–4 keV CCD/grating landscape. → `docs/soft-xray-hardware-survey`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
2. **Parameter-space sampling review.** Design principled prioritization across high-dimensional sweep parameters. → `docs/parameter-space-sampling-review`.
3. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
4. **pyelsepa / ELSEPA transport.** Maintain landed, validated adapter externally provisioned CI environment.
5. **Material filters.** Model calibration-filter transmission between x-ray beam and detector. → `feature/material-filters`.
6. **>user< Fix cached pull-in of pre-rendered animations in trace_app (add more buttons or something) -- maybe due to regenerate's changing random seed?**
   1. Progress bar here shows up at the top of the marimo notebook instead of near the button that is pressed to start the render -- confusing
   2. clip off the extra figure background & legend in the render, it is ugly. We just want the black grid space, with the colorbar and mat/config title info overlayed, but no background color. Saved render is also a bit pixelated, especially when opened in an mp4 viewer outside of the marimo app.
   3. add button to open render saving dialogue (so user can promptly move it from the cache)
7. **>user< CLI rework: `line-grid` → `energy-grid` + new `cxr sweep`.** Disentangle line-grid / brem-grid / derivation-geometry / scan-parameter-sweep settings; add read-write `cxr sweep` (default `[profiles.*]` + per-material scan ranges). Design: [`docs/cli-energy-grid-sweep-rework-plan.md`](docs/cli-energy-grid-sweep-rework-plan.md).
   1. Deferred within that rework: rename catalog `[profiles.*]` TOML table (scan-grid defaults) to `[scan_defaults.*]` to end the name clash with `profiles.py` `SweepProfile` (full/survey). Schema change — see P3.

## P3 - lower / exploratory / small bugfixes

1. **Rename catalog `[profiles.*]` TOML table → `[scan_defaults.*]`.** The scan-grid
   defaults table clashes name-wise with `profiles.py` `SweepProfile` (full/survey),
   two unrelated "profile" concepts. Catalog-schema change (materials.toml,
   `_parse_profiles`/`_parse_materials`, golden, docs). Blocked-by / folds into P2.7
   (`cxr sweep` rework). Design: [`docs/cli-energy-grid-sweep-rework-plan.md`](docs/cli-energy-grid-sweep-rework-plan.md).