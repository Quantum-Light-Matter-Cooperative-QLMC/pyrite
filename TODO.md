# TODO / Backlog

---

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
   `signed-off`, as are the four x-ray dispersion claims (`xray-chi-zero`,
   `xray-refractive-index`, `xray-in-medium-resonance`,
   `xray-in-medium-propagation-phase`) and `coherent-segment-midpoint-time`
   (electron-transport correctness fix, implementation landed on `main`).
   1. **Rederivation Clean-up and Formatting.** Need a worker to go thru and clean up the math in
      completed rederivations, just to make it human readable. Purely formatting, LateX, no logic change.
      Needed to facilitate sign-off. Related - make a ruleset on LateX/MyST formatting that is handed
      to fresh-context workers doing the rederivations.
2. **Energy-controlled electron transport.** Schema inventory, per-flight
   diagnostics, midpoint predictor-corrector stopping/clock, radiation error
   estimators, and convergence matrices are done (A-E): the binding control
   variable is absolute emission phase, not fractional energy loss, and the
   midpoint rule is a prerequisite for coherent CXR. Remaining: the
   energy-controlled propagator with collision optical-depth handling,
   CXR/bremsstrahlung invariance to substep refinement, porting the accepted
   algorithm to all four execution paths, and ledger/docs/golden-data closure
   (F-I). →
   `feature/energy-controlled-electron-transport`;
   [`agentdocs/tasks/feature/energy-controlled-electron-transport/`](agentdocs/tasks/feature/energy-controlled-electron-transport/).
3. **Readable physics validation ledger.** Replace the squeezed eight-column
   Sphinx table with one labeled record per validation ID, preserving all
   scientific content and generated summaries. →
   `docs/readable-physics-validation-ledger`;
   [`agentdocs/tasks/docs/readable-physics-validation-ledger/`](agentdocs/tasks/docs/readable-physics-validation-ledger/).
4. **Compute performance optimization — remainder.** Reopened 2026-08-14 on new
   evidence: `qlmc` job `1638` (`hopg_hbn`) computed 5508 hopg cases in 195 s and
   5508 h-BN cases in 206 s — ~35-37 ms/case — while the box held ~5% CPU and
   10-15% GPU utilization with no burstiness, so the case loop is bound by
   neither CPU nor GPU. Checkpoint I/O is no longer the suspect: after the
   record-table encoding (`543394b`) that job saved 5508 records per material in
   ~11 s and ~6 s, against the 152 s version-1 decodes that returned zero work in
   jobs `1635`/`1636` and were SIGTERM-killed in `1637`. Remaining work is what
   W3 item 2 and W4 already scoped and gated on a CUDA box: one authorized remote
   job exercising the ported `pyrite run -R --cpu`/`--cpu-only` profiler, NVTX
   ranges in `transport.py`, and attribution of the host-side transport remainder
   (`.capsync` was 28.5 ms of a 64.3 ms local hopg case). →
   `feature/compute-performance-optimization`;
   [`agentdocs/tasks/feature/compute-performance-optimization/REMAINING.md`](agentdocs/tasks/feature/compute-performance-optimization/REMAINING.md).

## P1 - top-priority back burner

### Ready

1. **Add support for characteristic X-rays**
2. **Debye-Waller provenance and anisotropy audit.** Replace placeholder/reused `B_ang2`; evaluate scalar sensitivity, tensor
   factors, and DFPT value/complexity. → `feature/debye-waller-audit`; [`docs/validation/materials/debye-waller-audit.md`](docs/validation/materials/debye-waller-audit.md). >user<
3. **Validate batched coherent line accumulation.** The implementation and
   `Validation: coherent-line-hkl-batch` ledger row landed; fresh-context
   verification and human sign-off remain. → `feature/coherent-line-batching`;
   [`agentdocs/tasks/feature/coherent-line-batching/`](agentdocs/tasks/feature/coherent-line-batching/).
4. **Physics boundary typing.** `typecheck` passes, but the physics packages are
   the least annotated in the tree — `montecarlo` 4.6% of params, `detectors`
   6.6% (vs `materials` 59.8%); of 42 public re-exported functions, 2 are fully
   annotated. Annotate the public boundary only, with unit-carrying aliases
   (278 `_ang` / 228 `_rad` / 206 `_eV` names carry units in spelling alone) and
   explicit `| None`. Not a blanket pass over array helpers; no stricter ty gate
   (`--error all` is 51 diagnostics, mostly correlated-guard false positives).
   → `chore/physics-boundary-typing`;
   [`agentdocs/tasks/chore/physics-boundary-typing/`](agentdocs/tasks/chore/physics-boundary-typing/).

### Gated

1. **Reference elastic scattering data.** Replace out-of-range Browning totals
   in `mott` mode with provenance-controlled NIST totals and adopt DCS CDFs or a
   validated higher-moment surrogate. Gated on redistribution and model-form
   decisions. → `feature/reference-elastic-scattering-data`;
   [`agentdocs/tasks/feature/reference-elastic-scattering-data/`](agentdocs/tasks/feature/reference-elastic-scattering-data/).
2. **Reference electron stopping data.** Package provenance-controlled
   collisional/radiative stopping over 1--300 keV, retain Joy--Luo as an
   explicit fallback, and expose model/fallback metadata. Gated on source,
   redistribution, compound, and low-energy splice decisions.
   → `feature/reference-electron-stopping-data`;
   [`agentdocs/tasks/feature/reference-electron-stopping-data/`](agentdocs/tasks/feature/reference-electron-stopping-data/).
3. **Measured-data validation.** General experimental-simulation comparison & validation. Particularly: compare modeled broadened line widths vs measured HOPG rocking-curve / EDS dataset. Design: [`docs/physics/materials/crystal-mosaicity.md`](docs/physics/materials/crystal-mosaicity.md).
4. **Superradiant PXR/CBS validation.** Optional phased segment/electron sum is implemented but unverified; resolve phase convention and bunch-form-factor limits before scientific use. Design: [`docs/physics/radiation-physics/coherent-emission.md`](docs/physics/radiation-physics/coherent-emission.md).

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
3. **Analysis handoff and figure reproducibility.** `pyrite app analysis export`
   renders default selector state regardless of the requested view
   (`apps/export.py:29` passes no `--` argv; the smoke path
   `apps/analyze.py:411` does), and exported HTML carries no machine-readable
   provenance even though `dataset_identity` is in hand at load time — the
   tutorial asks the human to transcribe the digest. Sequence: parameterise the
   export and stamp digest/stem/version/revision into the page; require explicit
   identity for `export` rather than falling back to persisted machine-local
   defaults; add `html-wasm` and plotted-series data output; then collapse
   launch, export, and `anchor_figures` onto one parameterised render entry
   point so interactive and publication figures cannot disagree. Review:
   [`agentdocs/plans/analysis-surface-review.md`](agentdocs/plans/analysis-surface-review.md)
   §4 G1/G2/G4/G6, §6.
4. **Machine-readable validation check records.** `checks/*.py` print
   `PASS`/`FAIL` to stdout, produce no artifact, sit outside the `test-suite`
   selectors, and map to ledger ids only through a hand-maintained table in
   `checks/README.md`. Have each check emit one JSON record (ledger id, measured
   value, reference value, tolerance, verdict, revision) and generate
   `docs/validation/status-summary.md` from those records instead of prose, so a
   passing check becomes dated, diffable evidence. Peers do this: Geant4 diffs
   example reference outputs in CI, geant-val stores every comparison. Supports
   ledger sign-off (Active 1). Review:
   [`agentdocs/plans/analysis-surface-review.md`](agentdocs/plans/analysis-surface-review.md)
   §4 G3.

## P3 - lower-priority / exploratory back burner

1. **Parameter-space sampling review.** Design principled prioritization across high-dimensional sweep parameters. → `docs/parameter-space-sampling-review`; proposal: [`docs/research/workflows/parameter-space-sampling.md`](docs/research/workflows/parameter-space-sampling.md).
2. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/research/instrumentation/grazing-grating.md`](docs/research/instrumentation/grazing-grating.md).
3. **Positioned photon filters and pixel-resolved detection.** Add standalone
   finite material filter plates with pose and analytic attenuation between the
   target and a planar pixel detector, including partial coverage without a
   general geometry engine. → `feature/positioned-photon-filters`;
   [`agentdocs/tasks/feature/positioned-photon-filters/`](agentdocs/tasks/feature/positioned-photon-filters/).
4. **Result interchange for microscopy tooling.** Optional
   `pyrite[interchange]` writer exporting per-case spectra to `.hspy`/`.emd`/
   `.nxs` via standalone RosettaSciIO, with the identity digest in the signal
   metadata, so microscopists can open PyRITE output beside measured data
   without installing PyRITE. One-way export, never a checkpoint replacement;
   only worth building after exported artifacts carry provenance (P2 3).
   Decision recorded: **HyperSpy itself is rejected as a dependency** — sweeps
   are sparse (99 of 108 cells in the measured `hopg` checkpoint) and the signal
   axis varies per navigation position (four distinct `E_grid` definitions, one
   per beam energy), which HyperSpy's dense single-signal-axis array model
   cannot express without NaN-padding and lossy resampling; it also costs 18 new
   packages and its `exspy` EDS layer duplicates the ledgered detector model
   empirically. Revisit xarray, not HyperSpy, only if `results/selection.py`
   keeps growing. Review:
   [`agentdocs/plans/analysis-surface-review.md`](agentdocs/plans/analysis-surface-review.md)
   §5.
5. **Verify documented code blocks.** `docs/conf.py` loads `myst_parser` and
   autodoc only, so every fenced example in `docs/guides/*.md` is unverified
   prose that rots silently; abTEM, OpenMC, and HyperSpy all execute their
   documented examples. Triage found only 4 of 49 guide blocks are `python`
   and 37 are `bash`, so a doctest pass would reach ~8%; the plan instead
   checks `pyrite`/`pyrite-dev` blocks against the live Click and argparse
   trees and covers the Python narrative separately. →
   `chore/verify-documented-code-blocks`;
   [`agentdocs/tasks/chore/verify-documented-code-blocks/`](agentdocs/tasks/chore/verify-documented-code-blocks/).
   Review:
   [`agentdocs/plans/analysis-surface-review.md`](agentdocs/plans/analysis-surface-review.md)
   §4 G5.

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

1. **Chunked checkpoint lifecycle and truthful progress.** Eliminate full checkpoint
   decode/rewrite stalls for cached and budget-paused remote sweeps, make shards a
   first-class resumable/pullable state, account setup/finalization in slice budgets,
   and report loading/computing/saving activity without stale "NOW TESTING" cases. →
   `fix/chunked-checkpoint-lifecycle`;
   [`agentdocs/tasks/fix/chunked-checkpoint-lifecycle/`](agentdocs/tasks/fix/chunked-checkpoint-lifecycle/).
2. **Result encoding and transfer overhead.** The version-1 HDF5 leaf encoding
   spends one HDF5 object (~840 B, ~125 us) on every Python scalar and mapping
   key, and `slim -o -` sends the resulting redundant metadata unframed: 6.5x
   the bytes and ~200x the CPU of the zstd-pickle format it replaced, measured
   both directions. Cost a 26-minute `pull --profile hopg_hbn` on 2026-08-14,
   and the same 95 s/component write stalls seen in Bugs 1. Fix both ends —
   schema version 2 as a columnar record table with content-addressed array
   blobs (measured 66x faster write, 89x faster read, 5.5x smaller), plus a
   whole-container zstd frame on the wire — while keeping ADR-0009's
   pickle-free, `h5py`-openable artifact and permanent version-1 reads.
   Supplies the A/B measurement Bugs 1 deferred.
   → `fix/result-encoding-overhead`;
   [`agentdocs/tasks/fix/result-encoding-overhead/`](agentdocs/tasks/fix/result-encoding-overhead/).
3. **Restore remote `pyrite run` cache flags and automatic pull.** Propagate
   `--no-cache` / `--recompute` through `pyrite run [PROFILE] -R` with their
   local semantics, then fix the error after successful remote attachment when
   checkpoints should auto-pull. →
   `fix/run-cache-flags-auto-pull`;
   [`agentdocs/tasks/fix/run-cache-flags-auto-pull/`](agentdocs/tasks/fix/run-cache-flags-auto-pull/).
4. Running `pyrite profile create <new_profile> --from <source_profile>` doesn't copy the
   source profile's materials list. Fix this, and check if other aspects are copied properly.
5. Related to below: when user runs `pyrite run <profile>`, but energy-grid bounds have
   not been derived for that case:
   1. Give them a suggestion on what to run. We could also add a flag that automatically runs
      the derivation for their energy + mat before running their actual profile.
   2. shorten up the lines that are spit out. currently, spits out list of ALL materials
      in the entire repo (even ones marked non-standard in mats_to_sim.toml) which need to be
      set up for this energy, along with a long boilerplate text string next to every material.

### Ergonomics (ship anytime)

1. Delete the stale root `notebooks/` tree — it now holds only `__pycache__`
   directories shadowing `analysis_ui` module names, left over from the move
   into `src/pyrite/apps/`.
2. **Checkpoint HDF5 suffixes.** Write new checkpoint payloads as `.h5`, not
   `line.pkl` / `brem.pkl`, while retaining reads of legacy `.pkl` payloads. →
   `fix/checkpoint-h5-suffix`;
   [`agentdocs/tasks/fix/checkpoint-h5-suffix/`](agentdocs/tasks/fix/checkpoint-h5-suffix/).

## Notebook backlog

1. Make it so 'narrow_auto' in make_axis_controls from `src/pyrite/apps/analysis_ui/controls.py`
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
- **Geometry interoperability.** Start from a concrete downstream consumer,
  define a format-neutral `PhotonSource` contract with normalization and
  closure tests, then select and validate an adapter; MCPL remains one
  candidate for particle-transport consumers. Arbitrary in-simulation geometry
  (multiple materials at arbitrary position, shape, and orientation; STL/STEP
  import) remains a recorded **non-goal** —
  [ADR-0008](docs/adr/0008-no-arbitrary-target-geometry.md).
- **Custom detector tooling.** Extend the existing detector model with CLI
  tools for defining, loading, saving, and editing detector responses,
  geometries, and resolution for photon and charged-particle detectors.
- **Support for atomic-scale (single-or-few alternating layer) heterostructures/superlattices**.
  CTR and/or PXR/CBS may produce interesting interference results when layers alternate at the atomic-scale.
