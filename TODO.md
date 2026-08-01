# TODO / Backlog

Priority-ordered backlog; authoritative on `main`. Branch copies are disposable
and auto-resolve to `main` on merge/rebase (`.gitattributes` `TODO.md merge=ours`
driver — run `uv run cxr-dev bootstrap` once per clone). Branch detail lives in
`tasks/<branch-name>/` (entry doc `README.md`); workflow and merge rules:
[`tasks/README.md`](tasks/README.md). `>user<` marks untriaged user text that
must remain until moved into a task file. Edit and drop items on `main`.


## P0 - >user< To be Triaged

1. Add graphitic carbon nitride as material
2. Fix broken materials project query (requires .env file with MP API Key)
3. Do another thorough CLI simplification/clarification sweep
   1. Why does `cxr profile` have `add` for all profile parameters except materials, which instead are `member`
   2. `completion` should probably be worked in with `cxr setup` somehow. Drop `completion install` to just `completion` also, unless we plan to add `uninstall` to remove the completions.
      1. is `completion install --shell <SHELL>` necessary, or can we autodetect/add to all present? Should we? or `install`/`remove` for all on default, allowing optional shell invocation when only one is desired? maybe that's already being done.
   3. for `cxr remote`: `prune`, `prune-jobs`, and `clear` need re-evaluation/clarification. Probably should be worked-in/in conjunction with the `checkpoint` rework that is in progress on a branch.
   4. Semi-related to above: stale `performance` logs currently have no way of being dropped and no automatic method to pull them
         1. Why do we need to manually write `--chunk-minutes=0` on perf log? why only one material allowed? If `--chunk-minutes=0` is req'd for good perf log, then auto-set it. If more than one mat is fine, drop the req, or emit a warning if somewhat problematic.
         2. Add the `-p` flag as an optional profile marker, so that some profiles will just automatically be run as perf tests?
4. Use `uv` workspaces to separate out CLI and maybe Analysis/App work from core computations? `uv` notes on their website that they use their own workspaces to separate out CLI in particular
   1. Either in concert with this, or separately, split up tests into subdirs (or whatever the standard pytest implementation is) to reduce the time spent running tests repo-wide before every single commit. (e.g., rendering tests don't need to be rerun when compute is changed, crystal config pulling doesn't need to be tested when notebooks are edited, etc.)
5. Set up comparisons/param sweeps to analyze effect of longitudinal bunch length & transverse bunch size on coherent bunching for otherwise identical bunch parameters
   1. tilted bunch front? maybe dumb, maybe easier way to get coherent enhancement?
6. Improve support for other shells/operating systems outside of WSL (especially agent hooks which run automatically)
   1. Create directory of shell scripts that are run as agent hooks to allow more complex setup & OS/shell handling
7. General src refactor (maybe following work in items above) to group related loose files in src/cxr_mc/ into appropriately scoped subdirs
8. Fix this: make '--profile' the default behavior, '-m/--material', etc
   1. (cxr-mc) ➜  cxr-mc git:(main) cxr remote pull --profile standard --all --level9
      Usage: cxr remote pull [OPTIONS] [STEM|MATERIAL@PROFILE]...
      Try 'cxr remote pull --help' for help.
      Error: pull --profile already selects the profile's materials; drop --all
      (cxr-mc) ➜  cxr-mc git:(main) cxr remote pull --profile standard --level9  
      Usage: cxr remote pull [OPTIONS] [STEM|MATERIAL@PROFILE]...
      Try 'cxr remote pull --help' for help.
      Error: profile 'standard' has no explicit material membership; name materials alongside --profile, or use --all
9. Related to above: 'cxr run standard' runs ALL materials rather than just those marked under [materials] in mats_to_sim.toml. The standard profile (and all profiles) should default their behavior to using [materials] unless explicitly set (this includes `pull`)
10. Laptop-local SLURM is set up, but `cxr run` isn't connecting to it.
11. Add local-version of `cxr remote clear` to drop local pickles. Same behavior as for remote.
12. LONG TERM GOALS (not now) (maybe a further applications for `uv` workspaces and/or some other way to split up optional packages):
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
5. **Cross-profile checkpoint case reuse.** Named catalog profiles get isolated
   checkpoint stems (`<material>@<profile>-<hash>`), so two profiles with
   overlapping cases (e.g. `sub_100keV`, `sub_200keV`) fully recompute shared
   cases instead of sharing them; investigate a dedup key/storage model.
   Investigation only, not yet scoped. → `feature/cross-profile-case-reuse`;
   [`tasks/feature/cross-profile-case-reuse/`](tasks/feature/cross-profile-case-reuse/).
6. **Checkpoint command surface rework.** Split `cxr clear` into scoped
   commands, fix or drop `cxr checkpoint recompute` under current profile
   conventions, and rework `cxr checkpoint` once shared-checkpoint provenance
   is decided. Blocked on `feature/cross-profile-case-reuse`'s dedup design;
   design-first. → `feature/checkpoint-command-rework`;
   [`tasks/feature/checkpoint-command-rework/`](tasks/feature/checkpoint-command-rework/).
7. **Dashboard live verbosity cycling.** Press `v` in the local run dashboard
   or the remote `attach`/`status -a` viewer to cycle detail level (0/`-v`/
   `-vv`) live, no restart; reuses the existing `_KeyListener` keypress
   machinery from the remote cancel keybind. → `feature/dashboard-verbosity-keybind`;
   [`tasks/feature/dashboard-verbosity-keybind/`](tasks/feature/dashboard-verbosity-keybind/).

## P3 - lower / exploratory / small bugfixes / on-hold

1. **Parameter-space sampling review.** Design principled prioritization across high-dimensional sweep parameters. → `docs/parameter-space-sampling-review`.
2. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **pyelsepa / ELSEPA transport.** Maintain landed, validated adapter externally provisioned CI environment.
4.  **Material filters.** Model calibration-filter transmission between x-ray beam and detector. → `feature/material-filters`.
5. **`--no-cache` for run/remote run.** Add a flag to `cxr run`/`cxr remote
   run` that skips loading the existing checkpoint pickle and recomputes/
   overwrites it; remote chunked-queue interaction needs a design decision
   first. → `feature/run-no-cache`;
   [`tasks/feature/run-no-cache/`](tasks/feature/run-no-cache/).
6. **`-p`/`--perf` should force `--no-cache`.** A `cxr run --perf` sample
   against a warm checkpoint skips already-resumed cases, silently thinning
   the performance profile; force `--no-cache` (and audit other `--perf`
   flag interactions for the same silent-mismatch failure mode). Blocked on
   `feature/run-no-cache` landing. → `feature/perf-flag-no-cache-defaults`;
   [`tasks/feature/perf-flag-no-cache-defaults/`](tasks/feature/perf-flag-no-cache-defaults/).
7. **Tab completion latency.** Shell completion for `cxr` often takes
   multiple seconds; likely SSH-bound remote completion timeout or
   process-startup overhead, not confirmed. → `feature/tab-completion-latency`;
   [`tasks/feature/tab-completion-latency/`](tasks/feature/tab-completion-latency/).
