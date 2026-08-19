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

## Triaged

### P1 - top-priority back burner

#### Ready

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
5. **Energy-loss straggling.** Transport is pure CSDA: `_dEds_compound_scalar`
   returns a mean loss rate and every core applies it deterministically, so the
   fluctuation about it is discarded by construction. `energy-step-convergence`
   already measures the cost — by Jensen's inequality straggling biases the
   *mean* arrival time, ~0.3 rad at 25 keV over 1 um at 1 keV photon energy,
   above the 0.1 rad tolerance the midpoint propagator reaches — so numerical
   precision has outrun the transport model there. Not a drop-in sampler: the
   same write-up records of order one inelastic event per flight in carbon at
   25 keV, so the per-flight regime is single-collision rather than Gaussian or
   even Vavilov, and unrestricted Joy--Luo already carries the mean of the hard
   Moller tail whose fluctuation a full Landau draw would restore. The blocking
   decision is therefore model form — restricted stopping power plus explicit
   hard inelastic events, versus unrestricted CSDA plus a straggling
   distribution — not the sampler. Straggling off stays bit-for-bit on all four
   cores. → `feature/energy-loss-straggling`;
   [`agentdocs/tasks/feature/energy-loss-straggling/`](agentdocs/tasks/feature/energy-loss-straggling/).

#### Gated

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

#### Paused / on hold

1. **High-energy electron/channeling support.** Start with REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter), JungFrau detector ~0.5–4.5 m from interaction point. USER QUESTION: What is rep rate?
2. **Bent Crystals (After add channeling + relativistic electrons)**

### P2 - medium-priority back burner

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
5. **General 3D geometry engine evaluation.** Survey candidate engines
   (in-house quadric/CSG region table, NVIDIA Warp, Celeritas ORANGE/VecGeom,
   Geant4/pyg4ometry, OpenMC CSG + DAGMC, Embree/OptiX, mesh/CAD authoring
   libraries) against the device, determinism, coherent-emission, packaging, and
   migration axes, separately for target/electron-transport geometry,
   post-emission instrument geometry, and visualization. Documentation and ADR
   outcome only; a bounded throwaway microbenchmark is authorized under
   `scratch/`, with CUDA timing on the lab box via `pyrite remote`. Reversing
   [ADR-0008](docs/adr/0008-no-arbitrary-target-geometry.md) is permitted, so a
   superseding ADR-0011 and a reaffirming amendment are both live outcomes; the
   implementation is a separate task either way. No present scientific driver,
   so candidates are scored against latent backlog drivers — bent crystals
   first (P1 paused item 2), then finite/irregular targets, multi-object scenes,
   and instrument geometry. CUDA-only is acceptable with its dpnp/SYCL cost
   recorded. → `docs/geometry-engine-evaluation`;
   [`agentdocs/tasks/docs/geometry-engine-evaluation/`](agentdocs/tasks/docs/geometry-engine-evaluation/).
6. **Compute performance optimization — remainder.** Reopened 2026-08-14 on new
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

### P3 - lower-priority / exploratory back burner

1. **Parameter-space sampling review.** Design principled prioritization across high-dimensional sweep parameters. → `docs/parameter-space-sampling-review`; proposal: [`docs/research/workflows/parameter-space-sampling.md`](docs/research/workflows/parameter-space-sampling.md).
2. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/research/instrumentation/grazing-grating.md`](docs/research/instrumentation/grazing-grating.md).
3. **Positioned photon filters and pixel-resolved detection.** Add standalone
   finite material filter plates with pose and analytic attenuation between the
   target and a planar pixel detector, including partial coverage without a
   general geometry engine. → `feature/positioned-photon-filters`;
   [`agentdocs/tasks/feature/positioned-photon-filters/`](agentdocs/tasks/feature/positioned-photon-filters/).

## General backlogs

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
7. Add consistent naming, labelling, and reporting of material, chemical name, crystal phase, and crystal cut

### CLI

1. Add support for user-definition of FilterPlate (material, dims [simple rect dims], position, orientation)
2. Add more clear support for user control of calculation numerics

### Bugs (fix + regression test)

1. Rerunning a recently run & pulled `pyrite run <profile> -R` on a profile with remotely cached cases will correctly skip the cached ones for rerun, but it will then re-pull ALL the cases, including those that are identical to the stems held locally.

### Ergonomics (ship anytime)

Nothing here yet.

### Notebooks

1. Make it so 'narrow_auto' in make_axis_controls from `src/pyrite/apps/analysis_ui/controls.py`
   auto-sets the x-axis lims according to the Min/Max x-values of the widest *line energy*
   grid being plotted. Right now it goes off the brem grid, which makes it run to 10's or 100's
   of keV, defeating the whole purpose of the *narrow* plot.
2. Fix validation app:
   1. Fix broken anchors

      1. Detector: Solid Angle integration PASSES (no fix)
      2. Mosaic: exact orientation average PASSES (no fix)
      3. Multilayer: absorption + transport FAILS
      4. Multilayer: escape + backscatter FAILS
      5. Multilayer: crystalline substrate radiation FAILS
3. When trying to run supplementary Zhai checks for currently non-existent cases (new azims), we just get this:
   "Zhai cache missing or stale: /home/alex/dev/pyrite/checkpoints/zhai_reproduction/zhai-supplement-v4-9ae6e14b73b038b27138.pkl;
   populate it with `pyrite run --preset zhai --remote`". Trying to then use the recompute remotely feature, we get:

```JOB
  SLURM     1665
  Host      qlmc
  Workload  Zhai reproduction
  Monitor   pyrite job attach 20260816-084051-40a80198
  Status    pyrite job status 20260816-084051-40a80198 -vv
  Logs      pyrite job logs 20260816-084051-40a80198 --follow
  Pull      pyrite remote pull --preset zhai  (after completion)
[E 260816 08:40:58 notification_utils:257] (error_id=837366ce-6440-4722-9f63-5bfd67a026c8) remote launch did not report a job id:
    + scp /tmp/tmp8rvx8pue/cxr_code.tgz qlmc:/tmp/cxr_code.tgz
    + ssh -n qlmc mkdir -p "/home/aamador/dev/pyrite" && cd "/home/aamador/dev/pyrite" && tar xzf /tmp/cxr_code.tgz &&...<truncated></truncated>
    + ssh -n qlmc R="/home/aamador/dev/pyrite/jobs/reservations"; J="20260816-084051-40a80198"; mkdir -p "$R"; claimed...<truncated></truncated>

    JOB 20260816-084051-40a80198 · SUBMITTED
      SLURM     1665
      Host      qlmc
      Workload  Zhai reproduction
      Monitor   pyrite job attach 20260816-084051-40a80198
      Status    pyrite job status 20260816-084051-40a80198 -vv
      Logs      pyrite job logs 20260816-084051-40a80198 --follow
      Pull      pyrite remote pull --preset zhai  (after completion)
[E 260816 08:40:58 notification_utils:257] (error_id=84e77df3-6663-4da1-931b-59c16760cbcd) An ancestor raised an exception (RuntimeError):`
```

### UI backlog

1. When a profile parameter is set, CLI should print out something along the lines of "< changed from <old_value(s))> to <new_value(s))>" if there were old values, otherwise just "< set to <new_value(s))>"
2. Clean up raw printed ssh commands shipped to remote unless a verbose flag is
   given; otherwise show a well-formatted explanation, e.g. `Pulling "Standard Performance Profile for MoS2" [progress bar + absolute progress]`.
3. Add optional tags to materials that marks them for inclusion/exclusion in profiles or sweeps, e.g., 'high-energy', '

### Docs

1. Add explanation of substep splitting for CSDA

### Tests

1. Fix the many failing intel SYCL tests

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
  [ADR-0008](docs/adr/0008-no-arbitrary-target-geometry.md); P2 item 5
  re-examines that non-goal against named engines without presuming reversal.
- **Custom detector tooling.** Extend the existing detector model with CLI
  tools for defining, loading, saving, and editing detector responses,
  geometries, and resolution for photon and charged-particle detectors.
- **Support for atomic-scale (single-or-few alternating layer) heterostructures/superlattices**.
  CTR and/or PXR/CBS may produce interesting interference results when layers alternate at the atomic-scale.
