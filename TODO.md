# TODO / Backlog

Kanban-style backlog; authoritative on `main`. `Active` contains only work in
progress. P1-P3 are prioritized back-burner queues: items there are backlog,
gated, or paused, never active. Move an item into `Active` when work starts and
back to its priority queue when paused. `Long-term plans` records unprioritized
future direction, not committed work.

Branch copies are disposable and auto-resolve to `main` on merge/rebase
(`.gitattributes` `TODO.md merge=ours` driver — run `uv run cxr-dev bootstrap`
once per clone). Branch detail lives in `tasks/<branch-name>/` (entry doc
`README.md`); workflow and merge rules: [`tasks/README.md`](tasks/README.md).
`>user<` marks untriaged user text that must remain until moved into a task
file. Edit and drop items on `main`.

## Active

1. **Simplify and clarify the CLI surface.** Unify profile mutation vocabulary,
   settle completion install/remove and shell detection, rework local and remote
   checkpoint/job cleanup around shared per-case CAS ownership, resolve or drop
   checkpoint recompute, and add a coherent performance-log lifecycle with
   justified automatic mode defaults. →
   `feature/cli-surface-simplification`;
   [`tasks/feature/cli-surface-simplification/`](tasks/feature/cli-surface-simplification/).
2. **Compute performance optimization.** Measure and improve stable remote
   CPU/GPU/RAM/VRAM utilization without OOM, keeping the GPU fed where evidence
   supports it. → `feature/compute-performance-optimization`;
   [`tasks/feature/compute-performance-optimization/`](tasks/feature/compute-performance-optimization/).
3. **Physics validation ledger.** Continue fresh-context re-derivations, add missing in-code `Validation: <id>` markers. Design: [`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md); method: [`docs/validation/README.md`](docs/validation/README.md).
4. **Detector profiles and Zhai validation modernization.** Add profile-owned
   detector geometry with a 90 degree standard default, then route maintained
   Zhai/literature comparisons through current detector, Sweep, and case APIs.
   → `feature/profile-observation-angle`;
   [`tasks/feature/profile-observation-angle/`](tasks/feature/profile-observation-angle/).
5. **Debye-Waller provenance and anisotropy audit.** Replace placeholder/reused `B_ang2`; evaluate scalar sensitivity, tensor factors, and DFPT value/complexity. → `feature/debye-waller-audit`; [`docs/debye-waller-audit.md`](docs/debye-waller-audit.md). >user<

## P1 - top-priority back burner

### Ready

1. **Fix Analysis Compare loading and quality selection.** Analyze each
   material checkpoint once for all three Compare plots, preserve persistent
   cache reuse, and correct or accurately report the ratio plot's unexpected
   material exclusions. → `fix/analysis-compare-loading-quality`;
   [`tasks/fix/analysis-compare-loading-quality/`](tasks/fix/analysis-compare-loading-quality/).

### Gated

1. **Measured-data validation.** General experimental-simulation comparison & validation. Particularly: compare modeled broadened line widths vs measured HOPG rocking-curve / EDS dataset. Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).

### Paused / on hold

1. **High-energy electron/channeling support.** Evaluate `Geant4` or similar for REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter), JungFrau detector ~0.5–4.5 m from interaction point. USER QUESTION: What is rep rate?
2.  **Bent Crystals (After add channeling + relativistic electrons)**
3. **Superradiant PXR/CBS validation.** Optional phased segment/electron sum is implemented but unverified; resolve phase convention and bunch-form-factor limits before scientific use. Design: [`docs/coherent-emission.md`](docs/coherent-emission.md).

## P2 - medium-priority back burner

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
## P3 - lower-priority / exploratory back burner

1. **Parameter-space sampling review.** Design principled prioritization across high-dimensional sweep parameters. → `docs/parameter-space-sampling-review`.
2. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **pyelsepa / ELSEPA transport.** Maintain landed, validated adapter externally provisioned CI environment.
4.  **Material filters.** Model calibration-filter transmission between x-ray beam and detector. → `feature/material-filters`.
5. **Tab completion latency.** Shell completion for `cxr` often takes
   multiple seconds; likely SSH-bound remote completion timeout or
   process-startup overhead, not confirmed. → `feature/tab-completion-latency`;
   [`tasks/feature/tab-completion-latency/`](tasks/feature/tab-completion-latency/).

## Inbox - >user< to be triaged

1. **Coherent-bunching parameter study.** Compare effects of longitudinal bunch
   length and transverse bunch size on coherent bunching while holding other
   bunch parameters fixed; explore tilted bunch fronts or other routes to
   coherent enhancement.
2. **Cross-platform agent hooks.** Improve shell and operating-system support
   beyond WSL, including a directory of hook scripts for more complex setup and
   platform-specific handling.
3. **Source-package organization.** Group related loose modules under
   `src/cxr_mc/` into appropriately scoped subpackages after current structural
   work settles.
4. **Local SLURM integration.** Make `cxr run` use the configured laptop-local
   SLURM installation.
5. **Local checkpoint clearing.** Add a local equivalent of `cxr remote clear`
   for deleting local checkpoint pickles with matching behavior.
6. **CPU Performance Flag** Add an optional flag (-c/--cpu) that enables CPU performance profiling, and a second (mutually exclusive) --cpu-only flag which ONLY profiles the CPU. This is because the CPU profiling is really slow, generally. Additionally, the current funcitonality launches the combined CPU/GPU profile first, which works normally, but after that finishes it starts the CPU-only task which is not properly tracked by the prgress dashboard (bug).
7. **Block command name reuse as object names** Maybe already implemented, but it seems a possibility that a user might unintentionally name a profile or some other object one of the command names unintentionally when misusing it, e.g., `cxr profile create set hopg` might create a profile named 'set'. Seems an easy-ish thing to just block outright, no duplicating command names to avoid confusion. Low priority.

## Long-term plans

Direction notes only; not prioritized backlog or active commitments.

- **Workflow TUI.** Explore a dedicated interface, likely using Textual, for
  navigating checkpoints, running commands and sweeps, editing profiles, and
  extending the progress dashboard.
- **Broader physics scope.** Generalize beyond CXR with optional physics across
  wider energy regimes. Possible directions include secondary-electron
  emission, material ionization, high-energy interactions, electron coherence
  and QED effects, coherent transition radiation, and transport of protons,
  ions, or neutrons.
- **Complex geometry and interoperability.** Support multiple physical
  materials with arbitrary position, shape, and orientation; research
  interoperability with established simulation and PIC tools such as WarpX;
  evaluate standard import/export formats, including STL and STEP geometry.
- **Custom detector tooling.** Extend the existing detector model with CLI
  tools for defining, loading, saving, and editing detector responses,
  geometries, and resolution for photon and charged-particle detectors.
