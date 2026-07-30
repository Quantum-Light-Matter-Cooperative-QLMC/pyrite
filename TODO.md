# TODO / Backlog

Priority-ordered backlog; authoritative on `main`. Branch copies are disposable
and auto-resolve to `main` on merge/rebase (`.gitattributes` `TODO.md merge=ours`
driver — run `uv run cxr-dev bootstrap` once per clone). Branch detail lives in
`tasks/<branch-name>/` (entry doc `README.md`); workflow and merge rules:
[`tasks/README.md`](tasks/README.md). `>user<` marks untriaged user text that
must remain until moved into a task file. Edit and drop items on `main`.

## P1 - top-priority / high-value

### Active

1. **Compute performance optimization.** Measure and improve stable remote
   CPU/GPU/RAM/VRAM utilization without OOM, keeping the GPU fed where evidence
   supports it. → `feature/compute-performance-optimization`;
   [`tasks/feature/compute-performance-optimization/`](tasks/feature/compute-performance-optimization/).
2. **Physics validation ledger.** Continue fresh-context re-derivations, add missing in-code `Validation: <id>` markers. Design: [`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md); method: [`docs/validation/README.md`](docs/validation/README.md).
3. **Detector profiles and Zhai validation modernization.** Add profile-owned
   detector geometry with a 90 degree standard default, then route maintained
   Zhai/literature comparisons through current detector, Sweep, and case APIs.
   → `feature/profile-observation-angle`;
   [`tasks/feature/profile-observation-angle/`](tasks/feature/profile-observation-angle/).
4. **Debye-Waller provenance and anisotropy audit.** Replace placeholder/reused `B_ang2`; evaluate scalar sensitivity, tensor factors, and DFPT value/complexity. → `feature/debye-waller-audit`; [`docs/debye-waller-audit.md`](docs/debye-waller-audit.md). >user<
5. **Coherent/incoherent emission profiles.** Move `--coherent/--incoherent`
   into a `SweepProfile.emission` policy (`incoherent|coherent|both`); a single
   transport computes both spectra for `both`; add an analysis-app emission
   selector reading the sidecar. Stem-text embedding is delegated to
   `feature/checkpoint-variant-naming`. → `feature/profile-emission-modes`;
   [`tasks/feature/profile-emission-modes/`](tasks/feature/profile-emission-modes/).
6. **Checkpoint variant naming + analyze-menu visibility.** Reconcile shipped
   `<material>--<fidelity>-<hash>` checkpoint stems with the locked
   `<material>@<profile>` plan (or amend the plan); fix `identity_from_stem`'s
   fragile re-hash lookup to read the `meta.json` sidecar instead. →
   `feature/checkpoint-variant-naming`;
   [`tasks/feature/checkpoint-variant-naming/`](tasks/feature/checkpoint-variant-naming/).

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
   [`tasks/feature/portable-gpu-backends/`](tasks/feature/portable-gpu-backends/).
2. **Backend auto-detection and first-run setup prompt.** Detect installed
   NVIDIA/AMD/Intel GPU hardware and offer a first-run CLI prompt to enable
   GPU acceleration (default CPU otherwise); persist the choice to `.env`'s
   `CXR_MC_BACKEND`, only when unset. → `feature/backend-autodetect`;
   [`tasks/feature/backend-autodetect/`](tasks/feature/backend-autodetect/).
3. **Grazing grating — ALEX-s constants + hardware survey.** Research cited device constants and ~10 eV–4 keV CCD/grating landscape. → `docs/soft-xray-hardware-survey`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
4. **Longitudinal bunch profiles and coherence comparison.** Add charge-matched
   200 fs Gaussian, wavelength-matched microbunch-train, and compressed-bunch
   HOPG/h-BN profiles with paired coherent/incoherent analysis. →
   `feature/longitudinal-bunch-profiles`;
   [`tasks/feature/longitudinal-bunch-profiles/`](tasks/feature/longitudinal-bunch-profiles/).
5. **Local run progress dashboard.** Replace the bare `cxr run <profile>` tqdm
   bar with the multi-panel dashboard used by `cxr remote status -a -vv`,
   rendering only the panels with local data — omit SLURM `SQUEUE` off-node and
   GPU rows without a GPU; fall back to tqdm when non-interactive. →
   `feature/local-run-dashboard`;
   [`tasks/feature/local-run-dashboard/`](tasks/feature/local-run-dashboard/).

## P3 - lower / exploratory / small bugfixes / on-hold

1. **Parameter-space sampling review.** Design principled prioritization across high-dimensional sweep parameters. → `docs/parameter-space-sampling-review`.
2. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **pyelsepa / ELSEPA transport.** Maintain landed, validated adapter externally provisioned CI environment.
4.  **Material filters.** Model calibration-filter transmission between x-ray beam and detector. → `feature/material-filters`.
