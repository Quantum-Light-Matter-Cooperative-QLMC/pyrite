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

**NOTE:** User wrote item straight into `TODO.md` (denoted >user<)? Fold into branch (steps 1-2), then drop >user< & slim to one-line summary on `main` once branch exist.

## P1 - top-priority/high-value

### Active

1. **Physics validation ledger.** Continue fresh-context re-derivations, add missing in-code `Validation: <id>` markers. Design: [`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md); method: [`docs/validation/README.md`](docs/validation/README.md).
2. **Grazing grating — ALEX-s constants + hardware survey.** Research cited device constants and ~10 eV–4 keV CCD/grating landscape. → `docs/soft-xray-hardware-survey`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **Checkpoint rework** split up material pickles into individual line and brem pickles, with each mat having a subdir in ./checkpoints/
4. **>user< Named sweep profiles and dataset identity.** Define independently configurable `full` (current production defaults) and provisional `survey` (coarser grids, narrower ranges, lower electron counts, and reduced angle/thickness/reflection sets) profiles; include profile plus resolved-parameter provenance in dataset identity, and design variant storage/archive handling with the checkpoint rework. Naming alternatives: `preview` or `coarse`.
5. **>user< Persistent component-recompute defaults and batch parity.** Extend existing line-only `reline` and brem-only `rebrem` commands with profile-aware start/stop/step/electron-count defaults shared across explicit material lists or `--all`; keep current names unless a unified component interface adds clear value, and add explicit multi-material partial-pull regression tests (`--brem-only`/`--line-only` already apply to the full material list).
6. **>user< Zhai/NIST DTSA-II bremsstrahlung validation and subtraction.** Add versioned external-background fixtures, normalization/provenance documentation, model comparison, and experimental fit/subtraction in the validation path; reuse existing `load_external_brem` ingestion instead of treating DTSA-II as one canonical NIST dataset.
7. **>user< Literature provenance ledger.** Create a canonical bibliography with stable source IDs for major physics and data claims, audit primary-source/license quality, and link validation-ledger rows plus in-code `Validation: <id>` markers without duplicating derivations.
8. **>user< Debye-Waller provenance and anisotropy audit.** Replace placeholder or reused `B_ang2` values with temperature/phase-specific primary-source values, record provenance in the validation/literature ledgers, and assess atom-specific or tensor `U` requirements before extending the scalar catalog schema.
9. **>user< `rebrem` GPU-memory regression.** Reproduce and measure `cxr remote rebrem --all --ne-brem 500 --step 20`; remote logs show VRAM accumulation after hundreds of records and current `_brem_for_case` bypasses the normal inter-case CuPy-pool release, so add a failing cadence regression, bounded-memory fix, and remote benchmark while preserving resumability.

### Gated

1. **Measured-data validation.** General experimental-simulation comparison & validation. Particularly: compare modeled broadened line widths vs measured HOPG rocking-curve / EDS dataset. Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).

### On-Hold

1. **High-energy electron/channeling support.** Evaluate `Geant4` or similar for REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter), JungFrau detector ~0.5–4.5 m from interaction point. USER QUESTION: What is rep rate?
2. **Bent Crystals (After add channeling + relativistic electrons)**
3. **Superradiant PXR/CBS** need bunch length knowledge, coherent emission *across segments* (also needed by channeling radiation as in long-term features #1, #3)

## P2 - medium-priority

1. **Add inv. lattice vector arrow(s) of interest to 3D Crystal Visualizer**
2. **Parameter-space sampling review.** Design principled prioritization across high-dimensional sweep parameters. → `docs/parameter-space-sampling-review`.
3. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
4. **pyelsepa / ELSEPA transport.** Maintain landed, validated adapter externally provisioned CI environment. Design: [`docs/cstool-nebula-evaluation.md`](docs/cstool-nebula-evaluation.md).
5. **Material filters.** Model calibration-filter transmission between x-ray beam and detector. → `feature/material-filters`.

## P3 - lower / exploratory

1. **>user< Pre-collaboration Git history cleanup.** Inventory published refs, choose a squash boundary and future commit policy, tag/back up current history, then coordinate any one-time force-push and reclone before additional contributors begin work.
