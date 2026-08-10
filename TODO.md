# TODO / Backlog

Kanban-style backlog; authoritative on `main`. `Active` contains only work in
progress. P1-P3 are prioritized back-burner queues: items there are backlog,
gated, or paused, never active. Move an item into `Active` when work starts and
back to its priority queue when paused. `Long-term plans` records unprioritized
future direction, not committed work.

Branch copies are disposable and auto-resolve to `main` on merge/rebase
(`.gitattributes` `TODO.md merge=ours` driver — run `uv run pyrite-dev bootstrap`
once per clone). Branch detail lives in `agentdocs/tasks/<branch-name>/` (entry
doc `README.md`); workflow and merge rules:
[`agentdocs/README.md`](agentdocs/README.md).
`>user<` marks untriaged user text that must remain until moved into a task
file. Edit and drop items on `main`.

## Active

1. **Physics validation ledger.** Continue fresh-context re-derivations, add missing in-code `Validation: <id>` markers. Design: [`docs/validation/physics-validation-ledger.md`](docs/validation/physics-validation-ledger.md); method: [`docs/validation/methodology.md`](docs/validation/methodology.md).
   The four incoherent-line-path markers are now ledgered (`line-hkl-batch`,
   `line-amplitude-fusion`, `line-gemv-elementwise` `filtered`;
   `line-absorption-tabulation` a `discrepancy` — tabulated `μ` off `2.72e-01`
   at hopg's C K-edge, unresolved). `beam-phase-space-injection` /
   `beam-energy-spread-injection` are `rederived` and await human
   `signed-off`.

## P1 - top-priority back burner

### Ready

1. **Electron transport correctness.** Pair coherent midpoint position with
   midpoint time, truncate cutoff-crossing flights, distinguish step-limited
   histories from physical stops, and tighten transport input validation across
   all CPU/CUDA cores. → `fix/electron-transport-correctness`;
   [`agentdocs/tasks/fix/electron-transport-correctness/`](agentdocs/tasks/fix/electron-transport-correctness/).
2. **Add support for characteristic X-rays**
3. **Debye-Waller provenance and anisotropy audit.** Replace placeholder/reused `B_ang2`; evaluate scalar sensitivity, tensor
   factors, and DFPT value/complexity. → `feature/debye-waller-audit`; [`docs/validation/materials/debye-waller-audit.md`](docs/validation/materials/debye-waller-audit.md). >user<
4. **Validate batched coherent line accumulation.** The implementation and
   `Validation: coherent-line-hkl-batch` ledger row landed; fresh-context
   verification and human sign-off remain. → `feature/coherent-line-batching`;
   [`agentdocs/tasks/feature/coherent-line-batching/`](agentdocs/tasks/feature/coherent-line-batching/).
5. **Physics boundary typing.** `typecheck` passes, but the physics packages are
   the least annotated in the tree — `montecarlo` 4.6% of params, `detectors`
   6.6% (vs `materials` 59.8%); of 42 public re-exported functions, 2 are fully
   annotated. Annotate the public boundary only, with unit-carrying aliases
   (278 `_ang` / 228 `_rad` / 206 `_eV` names carry units in spelling alone) and
   explicit `| None`. Not a blanket pass over array helpers; no stricter ty gate
   (`--error all` is 51 diagnostics, mostly correlated-guard false positives).
   → `chore/physics-boundary-typing`;
   [`agentdocs/tasks/chore/physics-boundary-typing/`](agentdocs/tasks/chore/physics-boundary-typing/).

### Gated

1. **Energy-controlled electron transport.** Introduce midpoint/integrated
   energy and time, controlled energy-loss/hazard substeps, physical-flight
   identity, and substep-invariant PXR/CBS/bremsstrahlung coupling. Gated on the
   correctness task and schema/convergence decisions.
   → `feature/energy-controlled-electron-transport`;
   [`agentdocs/tasks/feature/energy-controlled-electron-transport/`](agentdocs/tasks/feature/energy-controlled-electron-transport/).
2. **Reference elastic scattering data.** Replace out-of-range Browning totals
   in `mott` mode with provenance-controlled NIST totals and adopt DCS CDFs or a
   validated higher-moment surrogate. Gated on redistribution and model-form
   decisions. → `feature/reference-elastic-scattering-data`;
   [`agentdocs/tasks/feature/reference-elastic-scattering-data/`](agentdocs/tasks/feature/reference-elastic-scattering-data/).
3. **Reference electron stopping data.** Package provenance-controlled
   collisional/radiative stopping over 1--300 keV, retain Joy--Luo as an
   explicit fallback, and expose model/fallback metadata. Gated on source,
   redistribution, compound, and low-energy splice decisions.
   → `feature/reference-electron-stopping-data`;
   [`agentdocs/tasks/feature/reference-electron-stopping-data/`](agentdocs/tasks/feature/reference-electron-stopping-data/).
4. **Measured-data validation.** General experimental-simulation comparison & validation. Particularly: compare modeled broadened line widths vs measured HOPG rocking-curve / EDS dataset. Design: [`docs/physics/materials/crystal-mosaicity.md`](docs/physics/materials/crystal-mosaicity.md).
5. **Superradiant PXR/CBS validation.** Optional phased segment/electron sum is implemented but unverified; resolve phase convention and bunch-form-factor limits before scientific use. Design: [`docs/physics/radiation-physics/coherent-emission.md`](docs/physics/radiation-physics/coherent-emission.md).

### Paused / on hold

1. **High-energy electron/channeling support.** Start with REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter), JungFrau detector ~0.5–4.5 m from interaction point. USER QUESTION: What is rep rate?
2. **Bent Crystals (After add channeling + relativistic electrons)**

## P2 - medium-priority back burner

1. **Grazing grating — ALEX-s constants + hardware survey.** Research cited device constants and ~10 eV–4 keV CCD/grating landscape. → `docs/soft-xray-hardware-survey`; Design: [`docs/research/instrumentation/grazing-grating.md`](docs/research/instrumentation/grazing-grating.md).
2. **Detector profiles and Zhai validation modernization.** Add profile-owned
   detector geometry with a 90 degree standard default, then route maintained
   Zhai/literature comparisons through current detector, Sweep, and case APIs.
   → `feature/profile-observation-angle`;
   [`agentdocs/tasks/feature/profile-observation-angle/`](agentdocs/tasks/feature/profile-observation-angle/).

## P3 - lower-priority / exploratory back burner

1. **Parameter-space sampling review.** Design principled prioritization across high-dimensional sweep parameters. → `docs/parameter-space-sampling-review`; proposal: [`docs/research/workflows/parameter-space-sampling.md`](docs/research/workflows/parameter-space-sampling.md).
2. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/research/instrumentation/grazing-grating.md`](docs/research/instrumentation/grazing-grating.md).
3. **Material filters.** Model calibration-filter transmission between x-ray beam and detector..

## Inbox - >user< to be triaged

1. **Add capability to `--lock` and `--unlock` profiles to make them mutable/immutable**
2. **User-defined crystal cuts.** Allow user to define cut plane for given crystal, e.g., [110] or similar.
   If it is left undefined, cut should default to current default behavior.
3. **Coherent-bunching parameter study.** Compare effects of longitudinal bunch
   length and transverse bunch size on coherent bunching while holding other
   bunch parameters fixed; explore tilted bunch fronts or other routes to
   coherent enhancement.
4. **Cross-platform agent hooks.** Improve shell and operating-system support
   beyond WSL/Ubuntu, including a directory of hook scripts for more complex setup and
   platform-specific handling.
   1. Would be nice to add backend checking to install the right kind of
      extras to the agent (e.g., if computer has nvidia gpu, run
      `uv sync --all-groups --extra nvidia`, whereas if computer has intel gpu, then
      `uv sync --all-groups --extra intel`, etc.
      )
5. **Local SLURM integration.** Make `pyrite run` use the configured laptop-local
   SLURM installation (and, if available, report GPU usage statistics), if it exists.
6. **Add support for custom-defined remotes, or other standards for submitting remote computing tasks**.
   This will require research of what the industry standards/best practices are here, and we may find that the best practices are to leave it up to user custom scripting. If there are good standards for implementing comprehensive integrated support for SSH and/or other options for submitting, to SLURM or otherwise, write up a report on what we should do, why & how. This would obviously require more in depth capabilities for user configurations of their remote(s) of choice.

## CLI backlog

Nothing yet.

## Bugs (fix + regression test)

1. Running `pyrite profile create <new_profile> --from <source_profile>` doesn't copy the
   source profile's materials list. Fix this, and check if other aspects are copied properly.
2. Related to below: when user runs `pyrite run <profile>`, but energy-grid bounds have
   not been derived for that case:
   1. Give them a suggestion on what to run. We could also add a flag that automatically runs
      the derivation for their energy + mat before running their actual profile.
   2. shorten up the lines that are spit out. currently, spits out list of ALL materials
      in the entire repo (even ones marked non-standard in mats_to_sim.toml) which need to be
      set up for this energy, along with a long boilerplate text string next to every material.

### Ergonomics (ship anytime)

Nothing yet.

## Notebook backlog

1. Make it so 'narrow_auto' in make_axis_controls from notebooks/analysis_ui/controls
   auto-sets the x-axis lims according to the Min/Max x-values of the widest *line energy*
   grid being plotted. Right now it goes off the brem grid, which makes it run to 10's or 100's
   of keV, defeating the whole purpose of the *narrow* plot.

## UI backlog

1. Clean up raw printed ssh commands shipped to remote unless a verbose flag is
   given; otherwise show a well-formatted explanation, e.g. `Pulling "Standard Performance Profile for MoS2" [progress bar + absolute progress]`.
2. Golden data should be an optional installable, e.g. `uv add pyrite[golden]`
   or part of `uv add pyrite[all]`. Same with validation scripts/app -- not
   critically required.
3. Clarify that: 'gc' stands for garbage collection, rm stands for 'remove'.
   the `gc` help message could be a little more detailed, as well. functionality
   is a little unclear to me (if there is an identical case on both
   'standard' and 'sub_100keV' profiles, does `gc --profile standard` end up
   dropping the sub_100 keV cases too? I hope not.)

## Long-term plans

Direction notes only; not prioritized backlog or active commitments.

- **Workflow TUI.** Explore a dedicated interface, likely using Textual, for
  navigating checkpoints, running commands and sweeps, editing profiles, and
  extending the progress dashboard.
- **Broader physics scope.** Generalize beyond PXR/coherent brem with optional physics across
  wider energy regimes. Possible directions include electron energy-loss
  straggling and explicit hard inelastic events, consistent radiative stopping,
  direction-dependent bremsstrahlung, channeling as a separate advanced model,
  secondary-electron emission, material ionization, high-energy interactions,
  electron coherence and QED effects, coherent transition radiation, and
  transport of protons, ions, or neutrons. Near-term staging is tracked by the
  P1 electron-transport tasks above.
- **Complex geometry and interoperability.** Support multiple physical
  materials with arbitrary position, shape, and orientation; research
  interoperability with established simulation and PIC tools such as WarpX;
  evaluate standard import/export formats, including STL and STEP geometry.
- **Custom detector tooling.** Extend the existing detector model with CLI
  tools for defining, loading, saving, and editing detector responses,
  geometries, and resolution for photon and charged-particle detectors.
- **Support for atomic-scale (single-or-few alternating layer) heterostructures/superlattices**.
  CTR and/or PXR/CBS may produce interesting interference results when layers alternate at the atomic-scale.
