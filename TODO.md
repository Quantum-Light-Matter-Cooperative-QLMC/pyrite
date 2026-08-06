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

1. **Compute performance optimization.** Review cupyx.jit.rawkernel and Numba @njit
   optimizations are optimally executed, and review any critical physics changes. Generate new bit-for-bit or toleranced validation for the new paths where necessary. Ensure that non-nvidia fallbacks are present and functional (current tests should already be confirming this, but double check).
2. **Physics validation ledger.** Continue fresh-context re-derivations, add missing in-code `Validation: <id>` markers. Design: [`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md); method: [`docs/validation/README.md`](docs/validation/README.md).
3. **Debye-Waller provenance and anisotropy audit.** Replace placeholder/reused `B_ang2`; evaluate scalar sensitivity, tensor factors, and DFPT value/complexity. → `feature/debye-waller-audit`; [`docs/debye-waller-audit.md`](docs/debye-waller-audit.md). >user<

## P1 - top-priority back burner

### Ready

1. **Fix Analysis Compare loading and quality selection.** Analyze each
   material checkpoint once for all three Compare plots, preserve persistent
   cache reuse, and correct or accurately report the ratio plot's unexpected
   material exclusions. → `fix/analysis-compare-loading-quality`;
   [`tasks/fix/analysis-compare-loading-quality/`](tasks/fix/analysis-compare-loading-quality/).
2. **Beam phase space.** Activate the declared-inert `BeamSpec` transverse and
   energy-spread fields: canonical Twiss/normalized-emittance parameterization,
   per-electron injection sampling, profile decode, and finite initial phase
   space in the trajectory plots. The beam object and its profile plumbing
   already exist — this completes them. → `feature/beam-phase-space`;
   [`tasks/feature/beam-phase-space/`](tasks/feature/beam-phase-space/).
3. **Batched coherent line accumulation.** `coherent_emission=True` opts out of
   the batched `(n_seg, N_g)` line path and runs the legacy per-hkl loop: 38%
   of GPU-phase tottime on a profiled `hopg_coherent` run, ~2.65x slower per
   case than the batched branch. Physics-gated (reassociation → `Validation:`
   id, ledger row, golden regen). → `feature/coherent-line-batching`;
   [`tasks/feature/coherent-line-batching/`](tasks/feature/coherent-line-batching/).

### Gated

1. **Measured-data validation.** General experimental-simulation comparison & validation. Particularly: compare modeled broadened line widths vs measured HOPG rocking-curve / EDS dataset. Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).
2. **Superradiant PXR/CBS validation.** Optional phased segment/electron sum is implemented but unverified; resolve phase convention and bunch-form-factor limits before scientific use. Design: [`docs/coherent-emission.md`](docs/coherent-emission.md).

### Paused / on hold

1. **High-energy electron/channeling support.** Evaluate `Geant4` or similar for REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter), JungFrau detector ~0.5–4.5 m from interaction point. USER QUESTION: What is rep rate?
2. **Bent Crystals (After add channeling + relativistic electrons)**

## P2 - medium-priority back burner

1. **Grazing grating — ALEX-s constants + hardware survey.** Research cited device constants and ~10 eV–4 keV CCD/grating landscape. → `docs/soft-xray-hardware-survey`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
2. **Detector profiles and Zhai validation modernization.** Add profile-owned
   detector geometry with a 90 degree standard default, then route maintained
   Zhai/literature comparisons through current detector, Sweep, and case APIs.
   → `feature/profile-observation-angle`;
   [`tasks/feature/profile-observation-angle/`](tasks/feature/profile-observation-angle/).

## P3 - lower-priority / exploratory back burner

1. **Parameter-space sampling review.** Design principled prioritization across high-dimensional sweep parameters. → `docs/parameter-space-sampling-review`.
2. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **pyelsepa / ELSEPA transport.** Maintain landed, validated adapter externally provisioned CI environment.
4. **Material filters.** Model calibration-filter transmission between x-ray beam and detector..
5. **Tab completion latency.** Shell completion for `cxr` often takes
   multiple seconds; likely SSH-bound remote completion timeout or
   process-startup overhead, not confirmed. → `feature/tab-completion-latency`;
   [`tasks/feature/tab-completion-latency/`](tasks/feature/tab-completion-latency/).

## Inbox - >user< to be triaged

1. **User-defined crystal cuts.** Allow user to define cut plane for given crystal, e.g., [110] or similar.
   If it is left undefined, cut should default to current default behavior.
2. **Coherent-bunching parameter study.** Compare effects of longitudinal bunch
   length and transverse bunch size on coherent bunching while holding other
   bunch parameters fixed; explore tilted bunch fronts or other routes to
   coherent enhancement.
3. **Cross-platform agent hooks.** Improve shell and operating-system support
   beyond WSL/Ubuntu, including a directory of hook scripts for more complex setup and
   platform-specific handling.
      1. Would be nice to add backend checking to install the right kind of
         extras to the agent (e.g., if computer has nvidia gpu, run
         `uv sync --all-groups --extra nvidia`, whereas if computer has intel gpu, then
         `uv sync --all-groups --extra intel`, etc.
         )
4. **Source-package organization.** Group related loose modules under
   `src/cxr_mc/` into appropriately scoped subpackages after current structural
   work settles.
5. **Local SLURM integration.** Make `cxr run` use the configured laptop-local
   SLURM installation (and, if available, report GPU usage statistics), if it exists.
6. **Evaluate refactoring `monteccarlo/runner.py` and `montecarlo/transport.py`**
   into multiple smaller files.
7. **Add support for custom-defined remotes, or other standards for submitting remote computing tasks**.
   This will require research of what the industry standards/best practices are here, and we may find that the best practices are to leave it up to user custom scripting. If there are good standards for implementing comprehensive integrated support for SSH and/or other options for submitting, to SLURM or otherwise, write up a report on what we should do, why & how. This would obviously require more in depth capabilities for user configurations of their remote(s) of choice.

## CLI backlog

Command-surface bugs and ergonomics folded from the retired `TODO_CLI.md`. The
structural redesign (noun/verb ordering, artifact model, deprecation policy)
lives in [`docs/cli-redesign-rfc.md`](docs/cli-redesign-rfc.md) and its sub-RFCs,
not here.

### Bugs (fix + regression test)

1. Running `cxr profile create <new_profile> --from <source_profile>` doesn't copy the
   source profile's materials list. Fix this, and check if other aspects are copied properly.

### Ergonomics (ship anytime)

1. Add `[coherent|incoherent|both]` to `cxr profile set` (and `add`). If a user
   has individually added both, auto-switch to `both` — but make that switch
   explicit/logged, not implicit magic. `remove` can also be used, does the opposite
   (If on `both` and user `remove`'s `incoherent`, they explicitly get back `coherent`)
2. `cxr` with no args should print help, like `-h/--help`.

## Notebook backlog

1. Make it so 'narrow_auto' in make_axis controls from notebooks/analysis_ui/controls
   auto-sets the x-axis lims according to the Min/Max x-values of the widest *line energy*
   grid being plotted. Right now it goes off the brem grid, which makes it run to 10's or 100's
   of keV, defeating the whole purpose of the *narrow* plot.

## UI backlog

1. Clean up raw printed ssh commands shipped to remote unless a verbose flag is
   given; otherwise show a well-formatted explanation, e.g. `Pulling "Standard
   Performance Profile for MoS2" [progress bar + absolute progress]`.
2. Golden data should be an optional installable, e.g. `uv add cxr-mc[golden]`
   or part of `uv add cxr-mc[all]`. Same with validation scripts/app -- not
   critically required.

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
- **Support for atomic-scale (single-or-few alternating layer) heterostructures/superlattices**.
  CTR and/or PXR/CBS may produce interesting interference results when layers alternate at the atomic-scale.
