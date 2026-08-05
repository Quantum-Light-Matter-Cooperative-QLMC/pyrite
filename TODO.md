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
5. **CLI deprecation substrate (RFC D7).** Compatibility-warning harness for
   retired/renamed `cxr` spellings, prerequisite for
   [`docs/plans/cli-redesign-implementation-plan.md`](docs/plans/cli-redesign-implementation-plan.md)
   slice 3. → `feature/cli-deprecation-substrate`;
   [`tasks/feature/cli-deprecation-substrate/`](tasks/feature/cli-deprecation-substrate/).
6. **CLI verb collapse and module fold (RFC D4 / pkg P3).** Canonical
   `gc`/`rm`/`recompute` lifecycle verbs; retire `rebrem.py`/`reline.py`/`prune.py`
   into the checkpoint recompute/cleanup modules.
   [`docs/plans/cli-redesign-implementation-plan.md`](docs/plans/cli-redesign-implementation-plan.md)
   slice 4. → `feature/cli-verb-collapse-module-fold`;
   [`tasks/feature/cli-verb-collapse-module-fold/`](tasks/feature/cli-verb-collapse-module-fold/).

## P1 - top-priority back burner

### Ready

1. **Fix Analysis Compare loading and quality selection.** Analyze each
   material checkpoint once for all three Compare plots, preserve persistent
   cache reuse, and correct or accurately report the ratio plot's unexpected
   material exclusions. → `fix/analysis-compare-loading-quality`;
   [`tasks/fix/analysis-compare-loading-quality/`](tasks/fix/analysis-compare-loading-quality/).

### Gated

1. **Measured-data validation.** General experimental-simulation comparison & validation. Particularly: compare modeled broadened line widths vs measured HOPG rocking-curve / EDS dataset. Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).
2. **Superradiant PXR/CBS validation.** Optional phased segment/electron sum is implemented but unverified; resolve phase convention and bunch-form-factor limits before scientific use. Design: [`docs/coherent-emission.md`](docs/coherent-emission.md).

### Paused / on hold

1. **High-energy electron/channeling support.** Evaluate `Geant4` or similar for REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter), JungFrau detector ~0.5–4.5 m from interaction point. USER QUESTION: What is rep rate?
2. **Bent Crystals (After add channeling + relativistic electrons)**

## P2 - medium-priority back burner

1. **Portable GPU backends.** Add maintained non-NVIDIA accelerator support
   behind a vendor-neutral backend contract while preserving CUDA and NumPy.
   → `feature/portable-gpu-backends`;
   [`tasks/feature/portable-gpu-backends/`](tasks/feature/portable-gpu-backends/).
2. **Grazing grating — ALEX-s constants + hardware survey.** Research cited device constants and ~10 eV–4 keV CCD/grating landscape. → `docs/soft-xray-hardware-survey`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).

## P3 - lower-priority / exploratory back burner

1. **Parameter-space sampling review.** Design principled prioritization across high-dimensional sweep parameters. → `docs/parameter-space-sampling-review`.
2. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **pyelsepa / ELSEPA transport.** Maintain landed, validated adapter externally provisioned CI environment.
4. **Material filters.** Model calibration-filter transmission between x-ray beam and detector. → `feature/material-filters`.
5. **Tab completion latency.** Shell completion for `cxr` often takes
   multiple seconds; likely SSH-bound remote completion timeout or
   process-startup overhead, not confirmed. → `feature/tab-completion-latency`;
   [`tasks/feature/tab-completion-latency/`](tasks/feature/tab-completion-latency/).
6. **Add finite initial phase space (velocity vector spread) into 2D and 3D trajectory plots**

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
6. **Evaluate refactoring `monteccarlo/runner.py` and `montecarlo/transport.py`**
   into multiple smaller files, they are very long.

## CLI backlog

Command-surface bugs and ergonomics folded from the retired `TODO_CLI.md`. The
structural redesign (noun/verb ordering, artifact model, deprecation policy)
lives in [`docs/cli-redesign-rfc.md`](docs/cli-redesign-rfc.md) and its sub-RFCs,
not here.

### Bugs (fix + regression test)

1. `energy-line` / energy-grid azimuth accepts 0–360° instead of the physical
   `(90, 270)` limit.
2. `cxr [remote] run -p` requires BOTH a single material AND a profile —
   contradicts `run`'s own optional `-m`. Should accept profile, material, or
   both.
3. `cxr material` help text points to `cxr profile members`, which does not
   exist (membership is `profile set/add/remove --materials`). Stale pointer.
4. `cxr app analysis [MATERIAL] [COMMAND]` mixes an optional positional with a
   subcommand at the same level — a material named `export` collides with the
   `export` subcommand. (RFC D1 removes this structurally; live collision now.)

### Ergonomics (ship anytime)

1. Add `[coherent|incoherent|both]` to `cxr profile set` (and `add`). If a user
   has individually added both, auto-switch to `both` — but make that switch
   explicit/logged, not implicit magic. `remove` can also be used, does the opposite
   (If on `both` and user `remove`'s `incoherent`, they explicitly get back `coherent`)
2. `cxr` with no args should print help, like `-h/--help`.

## UI backlog

General UI items folded from the retired `TODO_UI.md`.

1. Clean up raw printed ssh commands shipped to remote unless a verbose flag is
   given; otherwise show a well-formatted explanation, e.g. `Pulling "Standard
   Performance Profile for MoS2" [progress bar + absolute progress]`.
2. Golden data should be an optional installable, e.g. `uv add cxr-mc[golden]`
   or part of `uv add cxr-mc[all]`.

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
