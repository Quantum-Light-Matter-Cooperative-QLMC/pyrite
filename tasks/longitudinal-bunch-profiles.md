# Longitudinal bunch profiles for coherent-emission comparisons

Branch: `feature/longitudinal-bunch-profiles`

TODO scope: expand the former P2 coherence ON-vs-OFF comparison into three
charge-matched HOPG/h-BN campaigns that differ only in longitudinal bunch
structure.

## Goal

Provide three named catalog profiles over the same 16-case material/geometry
grid:

- materials: HOPG and h-BN;
- electron energies: 30 and 100 keV;
- crystal polar tilt: 45 degrees;
- crystal azimuth: 180 degrees;
- thicknesses: 100 nm, 500 nm, 1 um, and 4 um;
- circular transverse beam: 100 um diameter;
- identical total bunch charge, repetition rate, transverse distribution, and
  all non-longitudinal simulation settings.

The profiles are:

1. a standard 200 fs Gaussian longitudinal bunch;
2. a 200 fs XFEL-style envelope containing wavelength-matched microbunches;
3. one compressed Gaussian bunch with the derived microbunch duration.

Each profile must support paired phased/coherent and incoherent runs without
checkpoint collisions, retaining both the PXR/CBS line spectrum and
bremsstrahlung background. Add the longitudinal distribution, derivation,
serialization, CLI, and physical scaling structures needed to make those
comparisons meaningful.

## Current evidence

- `BeamSpec` already owns transverse FWHM, RMS `bunch_length_fs`,
  `long_shape`, explicit `long_offsets_fs`, `bunch_charge_pc`, and
  `rep_rate_hz`.
- Catalog `[profiles.NAME.beam]` blocks can serialize Gaussian/uniform bunch
  lengths or a fixed offset array. They cannot express a case-dependent
  microbunch train or wavelength-derived compressed bunch.
- `_sample_bunch_offsets` requires one explicit offset per simulated electron;
  this makes catalog offsets depend on numerical macro-electron count and is
  unsuitable as the profile-level microbunch representation.
- `cxr run --coherent` already phases trajectory segments and electrons through
  `t0_ang`. The `coherent-emission` ledger row remains `unverified`, including
  the `k.r` versus `(k+g).r` phase convention.
- Existing coherent output is normalized by simulated `Ne`. Its `N^2` limiting
  test covers simulated emitters but does not establish physical scaling by
  `bunch_charge_pc / e`, nor invariance as macro-electron count changes.
- Current profile CLI edits scan grids and electron-count grids but does not
  expose the catalog beam block.
- HOPG/h-BN already have derived 30 and 100 keV photon grids.

## Provisional conventions and decisions

- Treat 45 degrees as `tilt_deg` and 180 degrees as `tilt_azim_deg`.
  Preserve default detector observation angle `theta_obs_deg=90`.
- Encode thicknesses as `[1000, 5000, 10000, 40000]` angstrom.
- Treat “100 um transverse beam diameter” as circular Gaussian FWHM:
  `transverse_fwhm_x_mm = transverse_fwhm_y_mm = 0.1`.
- Until user supplies source values, retain current common defaults:
  `bunch_charge_pc=1.0`, `rep_rate_hz=5000.0`. Equality across profiles is
  required even if these values change.
- Keep the three campaign profiles independent from runtime fidelity
  (`full`/`survey`). Profile names proposed:
  `hopg_hbn_gaussian_200fs`, `hopg_hbn_microtrain_200fs`, and
  `hopg_hbn_compressed_microbunch`.
- Derive microbunch timing per material/beam-energy case from the selected
  target line angular frequency, not from electron energy or the photon-grid
  upper bound. For a Gaussian microbunch,
  `|F(omega)|^2 = exp[-(omega sigma_t)^2]`; derive
  `sigma_t <= sqrt(-ln eta) / omega` from an explicit retained-coherence target
  `eta`. Derive train spacing from an integer number of target periods,
  `T = 2*pi/omega = h/E_target`.
- Store a declarative longitudinal distribution/policy in the profile. Resolve
  sampled offsets only after material, beam energy, geometry, target
  reflection, and numerical electron count are known.
- Model the train as a finite 200 fs envelope with periodic microbunch centers,
  finite per-microbunch width, and explicit modulation depth/jitter parameters.
  Do not materialize tens of thousands of bunch centers when direct indexed or
  distribution sampling is equivalent.
- Separate physical electron count `N_q = bunch_charge_pc * 1e-12 / e` from
  simulated macro-electron count. A physical coherent estimator must converge
  with increasing macro-electron count and reduce to the incoherent result when
  the form factor vanishes.
- Use paired `--coherent` and `--incoherent` runs of each profile; do not create
  six mostly duplicated profiles.

## Open decisions

These require explicit resolution and documentation before profile values are
called scientifically meaningful:

- Does “200 fs length” mean RMS sigma, intensity/current FWHM, or full envelope
  span? Existing `BeamSpec.bunch_length_fs` means RMS.
- What retained-coherence target `eta` defines “appropriate” microbunch
  duration: for example 0.5, 0.9, or another experiment-driven value?
- Which emitted line defines `E_target` when multiple configured reciprocal
  families contribute: pinned dominant basal reflection, strongest predicted
  line, or a user-selected line?
- Is train spacing exactly one target period, an integer harmonic, or supplied
  by a known XFEL modulation mechanism? What modulation depth and timing jitter
  should be assumed?
- Confirm common physical charge and repetition rate; provisional values are
  1 pC and 5 kHz.
- Confirm 100 um means Gaussian FWHM rather than hard-edge diameter or another
  width convention.
- The unresolved `coherent-emission`/`line-energy-dispersion` phase convention
  may block a wavelength-derived scientific claim. Resolve it or explicitly
  constrain profile status to exploratory.

## Owning paths

- Beam/distribution schema and case expansion:
  `src/cxr_mc/sweep.py`, `src/cxr_mc/config.py`
- Longitudinal sampling and transport clock:
  `src/cxr_mc/montecarlo/transport.py`
- Physical coherent estimator and spectrum execution:
  `src/cxr_mc/montecarlo/spectrum.py`,
  `src/cxr_mc/montecarlo/runner.py`
- Catalog profile parsing/data:
  `src/cxr_mc/materials/catalog.py`,
  `src/cxr_mc/data/materials.toml`
- Profile editing and public contracts:
  `src/cxr_mc/cli/profile.py`, `src/cxr_mc/cli/_catalog_io.py`,
  `docs/cli-reference.md`
- Dataset identity/checkpoint separation:
  `src/cxr_mc/profiles.py`, `src/cxr_mc/run.py`
- Paired analysis:
  `src/cxr_mc/results/`, `src/cxr_mc/plots/`,
  `notebooks/analysis_app.py`
- Physics evidence:
  `docs/coherent-emission.md`,
  `docs/physics-validation-ledger.md`,
  `docs/validation/`

## Implementation checklist

1. Freeze the duration, target-line, coherence-threshold, train-spacing,
   modulation, charge, repetition-rate, and transverse-width conventions.
   Record units and conversions.
2. Design a frozen, serializable longitudinal distribution model supporting
   Gaussian, microbunch-train, and wavelength-derived compressed modes.
   Preserve legacy `bunch_length_fs`/`long_shape` payload behavior or provide a
   deliberate migration with compatibility tests.
3. Add a pure target-line/timing derivation helper. Resolve against each
   material/energy/geometry case and selected reflection; return target photon
   energy, wavelength, period, microbunch RMS duration, and provenance.
4. Implement deterministic, seed-isolated sampling for the finite train and
   compressed bunch. Validate centering, envelope width, microbunch width,
   spacing, modulation/jitter, and independence from unrelated transport RNG
   streams.
5. Derive and implement macro-particle weighting for physical bunch charge.
   Separate incoherent self terms from coherent cross terms, document estimator
   assumptions, and demonstrate convergence/invariance against simulated
   electron count.
6. Thread the resolved distribution and derivation provenance through
   `BeamSpec`, catalog parsing, `build_cases`, runner payloads, performance
   metadata, result records, deterministic serialization, dataset identity,
   checkpoint pruning, and resume matching.
7. Extend `cxr profile create|set|add|show` only as needed to round-trip the
   longitudinal fields. Preserve existing semantics: `set` replaces supplied
   fields; `add` unions only genuinely set-like values and must not silently
   merge incompatible distribution definitions. Regenerate CLI reference.
8. Add the three named profiles with the exact shared grid and beam parameters.
   Ensure HOPG/h-BN membership is explicit and no other material is included.
9. Provide a paired coherent/incoherent comparison path reporting at least peak
   and integrated line-flux ratios while retaining bremsstrahlung and common
   charge/rate normalization.
10. Add source equation, assumptions, limiting cases, `Validation:` markers,
    ledger rows/updates, and independent fresh-context validation. Do not mark
    any row `signed-off`; human action only.
11. Run fast CPU verification locally. Run representative full/GPU campaigns
    only through `cxr remote`, after profile/math review.

## Acceptance checks

- Each profile resolves exactly 16 cases before coherent-mode pairing:
  2 materials x 2 electron energies x 4 thicknesses x 1 polar x 1 azimuth.
- A structured profile/case snapshot proves every field except longitudinal
  distribution is identical across all three profiles.
- Gaussian profile resolves the agreed 200 fs convention; train envelope
  resolves the same convention; compressed profile resolves the same
  case-specific microbunch duration used inside the train.
- Derived timing tests cover HOPG and h-BN at 30 and 100 keV, with explicit
  target reflection/energy provenance and `T=h/E_target`.
- Statistical/RNG tests cover Gaussian, train, and compressed sampling without
  coupling to transport draws.
- Coherent estimator tests cover one emitter, in-phase `N^2`, Gaussian form
  factor, train spectral comb, compressed limit, decoherent limit, charge
  scaling, and macro-electron convergence.
- `--coherent` and `--incoherent` identities/checkpoint stems cannot collide;
  resume/prune behavior remains exact.
- Profile create/set/add/show round-trips new fields and preserves confirmation,
  dry-run, JSON/help, replacement, and deduplication contracts.
- Catalog validation and material golden snapshot pass after profile additions.
- Focused suites include `tests/test_coherent_emission.py`,
  `tests/test_sweep.py`, `tests/test_profiles.py`,
  `tests/test_material_catalog.py`, profile CLI tests, checkpoint/run tests, and
  paired-result/plot tests.
- `scripts/dev.py lint`, `typecheck`, focused tests, CLI-reference check, and
  relevant documentation checks pass. Heavy simulations are not run locally.
- Fresh-context physics validation produces a write-up and ledger verdict;
  scientific-use wording remains gated until discrepancies are resolved and a
  human signs off.

## Non-goals

- Changing HOPG/h-BN crystallography, energy-grid bounds, detector geometry, or
  transverse/emittance physics unrelated to this comparison.
- Treating simulated macro-electron count as physical bunch charge.
- Claiming coherent enhancement from unvalidated phase or line-energy
  conventions.
- Running the full campaign on local WSL.

## Dispatch

Worker skill: `lead-task`

Required skills: `scientific-library`, `monte-carlo`, `physics-review`,
`physics-validation`, `regression-testing`, `cli-ui-ux`,
`documentation-maintenance`, `regen-golden`, `run-cxr-mc`, and
`remote-gpu-jobs` for full campaigns.

Suggested slices after conventions are approved:

1. fresh-context form-factor, target-period, and physical-charge derivation;
2. longitudinal schema/sampler/case serialization;
3. coherent estimator and convergence tests;
4. profile CLI/catalog/data plus generated docs/golden;
5. paired analysis and remote campaign verification.

Authority after dispatch: task-local checkpoint commits only; no push or TODO
writing unless separately granted. Delegation is appropriate for independent
physics validation and non-overlapping test/CLI slices.

Stop on unresolved duration/diameter/target-line conventions, phase-model
discrepancy affecting the derivation, macro-electron estimator dependence,
profile identity collisions, unrelated dirty work, or any request to infer
missing physical source parameters silently.
