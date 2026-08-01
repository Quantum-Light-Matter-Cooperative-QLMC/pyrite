# TODO / Backlog

Priority-ordered backlog; authoritative on `main`. Branch copies are disposable
and auto-resolve to `main` on merge/rebase (`.gitattributes` `TODO.md merge=ours`
driver — run `uv run cxr-dev bootstrap` once per clone). Branch detail lives in
`tasks/<branch-name>/` (entry doc `README.md`); workflow and merge rules:
[`tasks/README.md`](tasks/README.md). `>user<` marks untriaged user text that
must remain until moved into a task file. Edit and drop items on `main`.


## P0 - Active

1. **Partition packaging and tests with a uv workspace.** Evaluate and, where
   justified, separate core computation from CLI and optional analysis/app
   tooling while preserving imports, entry points, packaging, and full-suite
   coverage; add measured domain-focused test paths. →
   `feature/uv-workspace-split`;
   [`tasks/feature/uv-workspace-split/`](tasks/feature/uv-workspace-split/).
2. **Simplify and clarify the CLI surface.** Unify profile mutation vocabulary,
   settle completion install/remove and shell detection, rework local and remote
   checkpoint/job cleanup around shared per-case CAS ownership, resolve or drop
   checkpoint recompute, and add a coherent performance-log lifecycle with
   justified automatic mode defaults. →
   `feature/cli-surface-simplification`;
   [`tasks/feature/cli-surface-simplification/`](tasks/feature/cli-surface-simplification/).
## P0 - >user< To be Triaged

1. Fix broken materials project query (requires .env file with MP API Key)
2. Set up comparisons/param sweeps to analyze effect of longitudinal bunch length & transverse bunch size on coherent bunching for otherwise identical bunch parameters
   1. tilted bunch front? maybe dumb, maybe easier way to get coherent enhancement?
3. Improve support for other shells/operating systems outside of WSL (especially agent hooks which run automatically)
   1. Create directory of shell scripts that are run as agent hooks to allow more complex setup & OS/shell handling
4. General src refactor (maybe following work in items above) to group related loose files in src/cxr_mc/ into appropriately scoped subdirs
5. Laptop-local SLURM is set up, but `cxr run` isn't connecting to it.
6. Add local-version of `cxr remote clear` to drop local pickles. Same behavior as for remote.
7. LONG TERM GOALS (not now) (maybe a further applications for `uv` workspaces and/or some other way to split up optional packages):
   1. Make this repo less CXR-specific, more general purpose.
      1. Add more optional physics (300 keV & below, to start with)
         1. Secondary electron emission (as an option)
         2. Material ionization?
         3. Obviously the higher-energy stuff
         4. Research other effects worth including
         5. Electron coherence (QED)?
         6. Coherent transition radiaton?
         7. Other particles? Protons, ions, neutrons? Presumably this is a very deep hole
      2. Deeper support for complex shapes, add support for multiple physical materials for interaction with arbitrary location, shape, & orientation
         1. Support for interaction with other libraries -- Requires research into common tools, filestandards, etc.. GPT? PIC Codes (warpX, etc.)?
         2. Easy file export/standard data format. Necessary? Are there standards at all? Does anyone want this?
         3. Importing of stl/stp files to define objects
      3. Add project tools (CLI object definition, control, interaction) for defining custom detectors, allowing loading & saving of custom detector responses, geometries, resolution, etc. (already partially implemented as Detector object)
         1. X-ray/photon detectors
         2. Electron/charged particle detectors 
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
2. **Grazing grating — ALEX-s constants + hardware survey.** Research cited device constants and ~10 eV–4 keV CCD/grating landscape. → `docs/soft-xray-hardware-survey`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **Longitudinal bunch profiles and coherence comparison.** Add charge-matched
   200 fs Gaussian, wavelength-matched microbunch-train, and compressed-bunch
   HOPG/h-BN profiles with paired coherent/incoherent analysis. →
   `feature/longitudinal-bunch-profiles`;
   [`tasks/feature/longitudinal-bunch-profiles/`](tasks/feature/longitudinal-bunch-profiles/).
4. **Local run progress dashboard.** Replace the bare `cxr run <profile>` tqdm
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
5. **Tab completion latency.** Shell completion for `cxr` often takes
   multiple seconds; likely SSH-bound remote completion timeout or
   process-startup overhead, not confirmed. → `feature/tab-completion-latency`;
   [`tasks/feature/tab-completion-latency/`](tasks/feature/tab-completion-latency/).
