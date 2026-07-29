# TODO / Backlog

Shared, priority-ordered backlog. Keep identical on every branch. Branch detail
lives in `tasks/<branch-leaf>.md`; workflow and merge rules:
[`tasks/README.md`](tasks/README.md). `>user<` marks untriaged user text that
must remain until moved into a task file. Reconcile with `todo-sync`.

## P1 - top-priority / high-value

### Active

1. **Compute performance optimization.** Measure and improve stable remote
   CPU/GPU/RAM/VRAM utilization without OOM, keeping the GPU fed where evidence
   supports it. → `feature/compute-performance-optimization`;
   [`tasks/compute-performance-optimization.md`](tasks/compute-performance-optimization.md).
2. **Physics validation ledger.** Continue fresh-context re-derivations, add missing in-code `Validation: <id>` markers. Design: [`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md); method: [`docs/validation/README.md`](docs/validation/README.md).
3. **Detector profiles and Zhai validation modernization.** Add profile-owned
   detector geometry with a 90 degree standard default, then route maintained
   Zhai/literature comparisons through current detector, Sweep, and case APIs.
   → `feature/profile-observation-angle`;
   [`tasks/profile-observation-angle.md`](tasks/profile-observation-angle.md).
4. **Debye-Waller provenance and anisotropy audit.** Replace placeholder/reused `B_ang2`; evaluate scalar sensitivity, tensor factors, and DFPT value/complexity. → `feature/debye-waller-audit`; [`docs/debye-waller-audit.md`](docs/debye-waller-audit.md). >user<

### Gated

1. **Measured-data validation.** General experimental-simulation comparison & validation. Particularly: compare modeled broadened line widths vs measured HOPG rocking-curve / EDS dataset. Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).

### On-Hold

1. **High-energy electron/channeling support.** Evaluate `Geant4` or similar for REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter), JungFrau detector ~0.5–4.5 m from interaction point. USER QUESTION: What is rep rate?
2.  **Bent Crystals (After add channeling + relativistic electrons)**
3. **Superradiant PXR/CBS validation.** Optional phased segment/electron sum is implemented but unverified; resolve phase convention and bunch-form-factor limits before scientific use. Design: [`docs/coherent-emission.md`](docs/coherent-emission.md).

## P2 - medium-priority

1. **Portable GPU backends.** Add maintained non-NVIDIA accelerator support
   behind a vendor-neutral backend contract while preserving CUDA and NumPy.
   → `feature/portable-gpu-backends`;
   [`tasks/portable-gpu-backends.md`](tasks/portable-gpu-backends.md).
2. **Grazing grating — ALEX-s constants + hardware survey.** Research cited device constants and ~10 eV–4 keV CCD/grating landscape. → `docs/soft-xray-hardware-survey`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **Longitudinal bunch profiles and coherence comparison.** Add charge-matched
   200 fs Gaussian, wavelength-matched microbunch-train, and compressed-bunch
   HOPG/h-BN profiles with paired coherent/incoherent analysis. →
   `feature/longitudinal-bunch-profiles`;
   [`tasks/longitudinal-bunch-profiles.md`](tasks/longitudinal-bunch-profiles.md).

## P3 - lower / exploratory / small bugfixes / on-hold

1. **Parameter-space sampling review.** Design principled prioritization across high-dimensional sweep parameters. → `docs/parameter-space-sampling-review`.
2. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **pyelsepa / ELSEPA transport.** Maintain landed, validated adapter externally provisioned CI environment.
4.  **Material filters.** Model calibration-filter transmission between x-ray beam and detector. → `feature/material-filters`.
