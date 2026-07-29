# TODO / Backlog

Shared, priority-ordered backlog. Keep identical on every branch. Branch detail
lives in `tasks/<branch-leaf>.md`; workflow and merge rules:
[`tasks/README.md`](tasks/README.md). `>user<` marks untriaged user text that
must remain until moved into a task file. Reconcile with `todo-sync`.

## P1 - top-priority / high-value

### Active

1. **>user< Compute Performance Optimization** Evaluate cause of low CPU/Host RAM Utilization + Bursty GPU utilization on remote lab box, look to improve compute to maximum stable (non-OOM) state -- roughly 85% CPU/GPU/RAM/VRAM utilization across the board, keep GPU always fed where possible.
2. **>user< SLURM Queue info** On `cxr remote attach` and related progress dashboard information (not in logs), want report of that job's position in the SLURM queue, along with top item in queue if job is pending.
3. **>user< Time estimate on compute progress bar** Next to the percent complete indicator on the compute progress bar in the progress dashboards, add an overall elapsed time/remaining time/total time estimate based on how long the current compute has taken and the fraction of currently-completed compute. 
4. **Physics validation ledger.** Continue fresh-context re-derivations, add missing in-code `Validation: <id>` markers. Design: [`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md); method: [`docs/validation/README.md`](docs/validation/README.md).
5. **Debye-Waller provenance and anisotropy audit.** Replace placeholder/reused `B_ang2`; evaluate scalar sensitivity, tensor factors, and DFPT value/complexity. → `feature/debye-waller-audit`; [`docs/debye-waller-audit.md`](docs/debye-waller-audit.md). >user<

### Gated

1. **Measured-data validation.** General experimental-simulation comparison & validation. Particularly: compare modeled broadened line widths vs measured HOPG rocking-curve / EDS dataset. Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).

### On-Hold

1. **High-energy electron/channeling support.** Evaluate `Geant4` or similar for REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter), JungFrau detector ~0.5–4.5 m from interaction point. USER QUESTION: What is rep rate?
2.  **Bent Crystals (After add channeling + relativistic electrons)**
3. **Superradiant PXR/CBS validation.** Optional phased segment/electron sum is implemented but unverified; resolve phase convention and bunch-form-factor limits before scientific use. Design: [`docs/coherent-emission.md`](docs/coherent-emission.md).

## P2 - medium-priority

1. **Grazing grating — ALEX-s constants + hardware survey.** Research cited device constants and ~10 eV–4 keV CCD/grating landscape. → `docs/soft-xray-hardware-survey`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
2. **Coherence ON-vs-OFF comparison.** Pair otherwise-identical coherent/incoherent datasets; report peak and integrated-flux ratios. Design: [`docs/coherent-emission.md`](docs/coherent-emission.md).

## P3 - lower / exploratory / small bugfixes / on-hold

1. **Parameter-space sampling review.** Design principled prioritization across high-dimensional sweep parameters. → `docs/parameter-space-sampling-review`.
2. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **pyelsepa / ELSEPA transport.** Maintain landed, validated adapter externally provisioned CI environment.
4.  **Material filters.** Model calibration-filter transmission between x-ray beam and detector. → `feature/material-filters`.