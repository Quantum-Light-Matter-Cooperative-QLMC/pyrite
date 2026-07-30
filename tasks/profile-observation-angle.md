# Detector profiles and Zhai validation modernization

Branch: `feature/profile-observation-angle`

TODO scope: add profile-owned detector configuration, default the standard
profile to 90 degrees observation angle, and migrate Zhai/literature
comparisons onto current configuration, case, and detector plumbing.

## Goal

Introduce one detector configuration object that owns observation geometry and
future detector-response metadata. Resolve it through catalog profiles,
`material_sweep`, case construction, dataset identity, and profile CLI. Then
make every maintained Zhai comparison use that same object and the current
simulation pipeline.

This is one task because the literature migration is the end-to-end proof that
nonstandard observation geometry survives profile resolution without a second
source of truth.

## Current evidence

- `Sweep` currently stores `theta_obs_deg`, `dtheta_obs_deg`, and `domega_sr`
  as unrelated flat fields. `build_cases` converts them to
  `theta_obs_rad`, `dtheta_obs_rad`, and `domega_sr`.
- `BeamSpec` intentionally owns only incident electron phase space and source
  properties. Detector geometry does not belong there.
- `material_sweep(..., theta_obs_deg=90.0)` hard-codes the default outside the
  catalog profile. `[profiles.NAME.beam]` is the only typed profile sub-block.
- `dataset_identity` serializes `Sweep`; adding a nested object naively would
  change every standard-profile digest and checkpoint stem.
- `checks/anchor_figures.py` keeps published Zhai geometry in separate
  radian-valued fields: 119 degree observation angle, 16.6 degree full polar
  span, and 0.066 sr. Fig. 1c and supplementary paths directly call transport
  and spectrum helpers instead of consistently resolving current Sweep/case
  configuration.
- `checks/feranchuk_vs_zhai_check.py` duplicates an older comparison pipeline.
  `notebooks/validation_app.py` calls the newer anchor module, while
  `cxr_mc._entry.reproduce_zhai` and remote validation populate its caches.

## Decisions

- Add frozen `DetectorSpec` in `src/cxr_mc/detectors/spec.py`; export it through
  the detector package. `Sweep.detector` becomes the canonical runtime owner.
- Active v1 fields:
  - `observation_angle_deg: float = 90.0`;
  - `polar_acceptance_deg: float | None = None`, defined as the full polar span
    and mapped to existing `dtheta_obs_rad`;
  - `solid_angle_sr: float | None = None`.
- Reserve optional, serializable detector-description fields without inventing
  hardware values: `response_model`, `qe_curve`, `pixel_pitch_um`,
  `sensor_thickness_um`, `distance_mm`, and `threshold_eV`. `None` means
  unspecified/current downstream default. These fields remain inert until a
  named detector adapter consumes them.
- Store profile configuration under `[profiles.NAME.detector]`. The standard
  profile explicitly sets `observation_angle_deg = 90.0`; old profiles lacking
  the block inherit the standard detector configuration.
- Do not create a one-field observation-angle class and do not add detector
  fields to `BeamSpec`.
- Preserve legacy flat `Sweep(theta_obs_deg=..., dtheta_obs_deg=...,
  domega_sr=...)` construction during migration. Normalize it to one
  `DetectorSpec`; reject conflicting flat and nested values instead of silently
  choosing.
- Resolution precedence:
  explicit `material_sweep`/runtime override, selected profile detector block,
  standard profile detector block, then the legacy 90 degree fallback.
- Keep existing case payload names and radian units. Downstream Monte Carlo
  kernels remain unchanged unless the migration exposes a real mismatch.
- Scalar detector fields use replace semantics. `cxr profile create`, `set`,
  and `show` expose active fields; `add`/`remove` remain for set-like grids and
  do not pretend scalar detector values are unionable.
- Zhai configuration uses one `DetectorSpec(119.0, 16.6, 0.066)` source.
  Analytic line energy, tilted geometry, line/bremsstrahlung spectra,
  broadening, display metadata, and cache identity all resolve from it.
- QE curves are portable registry/resource identifiers, not embedded arrays or
  machine-specific absolute paths.

## Owning paths

- Detector configuration and compatibility:
  `src/cxr_mc/detectors/spec.py`, `src/cxr_mc/detectors/__init__.py`,
  `src/cxr_mc/sweep.py`
- Profile decoding and resolution:
  `src/cxr_mc/materials/catalog.py`, `src/cxr_mc/data/materials.toml`,
  `src/cxr_mc/config.py`, `src/cxr_mc/profiles.py`
- Profile CLI and generated contract:
  `src/cxr_mc/cli/profile.py`, `src/cxr_mc/cli/_catalog_io.py`,
  `docs/cli-reference.md`
- Literature comparisons and app/runtime entry points:
  `checks/anchor_figures.py`, `checks/feranchuk_vs_zhai_check.py`,
  `notebooks/validation_app.py`, `src/cxr_mc/_entry/reproduce_zhai.py`,
  `src/cxr_mc/check.py`, `src/cxr_mc/_remote/`
- Tests:
  `tests/test_material_catalog.py`, `tests/test_profiles.py`,
  `tests/test_sweep.py`, profile CLI tests, `tests/test_anchor_figures.py`,
  `tests/test_reproduce_zhai.py`, `tests/test_check.py`,
  `tests/test_remote.py`

## Implementation checklist

1. Add validated `DetectorSpec` with finite/range checks, full-span acceptance
   semantics, immutable portable QE identifier, and documented inert fields.
2. Add `Sweep.detector`; migrate internal construction and case expansion.
   Provide flat-field compatibility and conflict errors at the public boundary.
3. Decode `[profiles.NAME.detector]`; explicitly add the 90 degree standard
   block and inherit it for legacy/custom profiles without detector settings.
4. Resolve detector precedence in `material_sweep`. Ensure explicit overrides
   are distinguishable from omission.
5. Preserve historical standard-profile serialized payload and
   `parameter_sha256`. Include only nondefault/new detector values in identity,
   so changed observation geometry cannot collide with 90 degree checkpoints.
6. Extend profile `create`, `set`, and `show` text/JSON/help contracts for
   active detector fields. Keep scalar replacement separate from grid
   `add`/`remove`; preserve standard-profile confirmation and `--dry-run`.
7. Regenerate CLI reference and material-catalog golden snapshot after catalog
   schema/data changes.
8. Refactor Zhai Fig. 1c and supplementary configuration to own one
   `DetectorSpec`. Build current `BeamSpec`/`Sweep` cases and use current runner
   or component helpers where scientifically equivalent; retain published
   literature inputs explicitly.
9. Route line-energy theory, geometry, coherent spectrum, bremsstrahlung,
   aperture convolution, plots, app labels, headless reproduction, and remote
   cache population through the resolved detector/case values.
10. Replace `feranchuk_vs_zhai_check.py` duplicate setup with canonical anchor
    helpers, or retire unsupported duplication while retaining its useful
    analytic-vs-Monte-Carlo diagnostic.
11. Bump/version Zhai cache identity so old caches cannot mask the migration.
    Store resolved detector metadata with newly generated artifacts.
12. Update tests and validation documentation only where ownership or
    provenance changed. No physics status advances and no `signed-off` edits.
13. Run fast CPU verification locally. Run expensive Zhai populations only
    through `cxr remote`.

## Acceptance checks

- `standard` resolves `DetectorSpec.observation_angle_deg == 90.0`; built cases
  carry `theta_obs_rad == pi/2`.
- A temporary/custom profile round-trips 119 degrees, 16.6 degrees full polar
  acceptance, and 0.066 sr through catalog load, CLI show/JSON, Sweep, cases,
  and dataset identity.
- Explicit runtime detector values override profile values. Omission inherits;
  conflicting legacy-flat and nested inputs fail with an actionable error.
- Reserved hardware fields validate and serialize when set but do not alter
  spectra until a response adapter explicitly consumes them.
- Existing standard-profile identity golden remains bit-for-bit unchanged.
  Any nondefault detector field changes the digest/checkpoint stem.
- Every maintained Zhai path uses 119 degrees from the resolved
  `DetectorSpec`; no independent hard-coded angle reaches line energy,
  geometry, spectrum, broadening, plot labels, or cache keys.
- Zhai line-energy and geometry tests prove that changing observation angle
  changes the result and that the published 119 degree configuration is used.
- Cache-schema tests reject/recompute artifacts created before detector
  metadata joined the key.
- `checks/feranchuk_vs_zhai_check.py`, validation app, headless reproducer, and
  remote job command agree on configuration provenance and cache locations.
- Focused catalog/profile/Sweep/CLI/Zhai tests pass. `marimo check` and
  `cxr app validation` smoke behavior pass. CLI reference and catalog golden
  checks pass.
- Full Zhai Monte Carlo is not run locally; remote execution is a dispatch-time
  validation step.

## Non-goals

- Implementing generic QE interpolation, pixel charge transport, or a new
  detector-response kernel.
- Replacing existing Timepix, Eagle XO, grating, or EDS response models.
- Changing published Zhai detector geometry or tuning it to improve agreement.
- Re-deriving or advancing physics-ledger claims as part of configuration
  migration.
- Running heavy validation sweeps on local WSL.

## Dispatch

Worker skill: `lead-task`

Required skills: `scientific-library`, `cli-ui-ux`, `monte-carlo`,
`regression-testing`, `notebook-workflow`, `documentation-maintenance`,
`regen-golden`, `run-cxr-mc`, `physics-review`, and `remote-gpu-jobs` for heavy
cache generation.

Suggested slices:

1. `DetectorSpec`, Sweep compatibility, profile decoding, and identity;
2. profile CLI, generated documentation, catalog golden, and fast contracts;
3. Zhai anchor/case/cache migration plus validation app and remote plumbing;
4. independent configuration/physics review and remote reproduction.

## Milestone 1 evidence

- Added frozen, validated `DetectorSpec`; reserved response/hardware fields
  serialize into dataset identity but remain absent from case physics payloads.
- `Sweep.detector` now owns runtime detector configuration. Legacy flat
  constructor inputs normalize into it; mixed conflicting inputs fail.
- Catalog profiles decode detector blocks with selected-profile → `standard` →
  90 degree fallback precedence. Bundled `standard` is explicit.
- `material_sweep` resolves profile detector defaults and explicit runtime
  overrides. Cases retain historical radian keys and full-span acceptance
  mapping.
- Historical `hopg`/`full` standard identity remains
  `d0bb205f2268b8cd30801b1146de8daf7e745ca70399919718519542a7c9b45c`;
  every nondefault detector field changes identity.
- Local evidence: 180 focused tests and 97 neighboring detector/config/run
  tests passed; catalog golden regenerated with no diff; lint and typecheck
  passed. Full CPU suite: 2070 passed, 39 skipped, 5 unrelated failures
  (agent-skill mirror drift, missing remote `attach` on branch, sandboxed
  forkserver permission).

Task-local checkpoint commits are allowed after dispatch. No worker may push,
edit canonical TODO ownership, mark physics `signed-off`, or run heavy work
locally without separate authority.

Stop on an unresolved identity collision, ambiguous acceptance-angle
convention, incompatible Zhai literature input, detector fields affecting
physics without an owning response model, or unrelated work outside explicit
paths.

## Milestone 2 evidence

- Profile `create`/`set` accept replace-only detector scalars; `show` text/JSON
  emits resolved active geometry. Standard edits confirm, `--yes` bypasses,
  and `--dry-run` never prompts.
- Added one maintained Zhai source:
  `DetectorSpec(119.0, 16.6, 0.066)` in `src/cxr_mc/_zhai.py`.
  Fig. 1c and supplementary paths derive radian case values from current
  `BeamSpec`/`Sweep`/`build_cases` construction.
- Analytic line energy, tilted geometry, PXR/CBS, bremsstrahlung, aperture
  broadening, detected-flux scaling, figure labels, headless reproduction, and
  remote job metadata resolve from that detector/case provenance.
- Zhai caches now use the `cxr.zhai-cache.v4` envelope and v4 filenames. Each
  record persists detector spec plus resolved historical case fields; raw,
  pre-detector, wrong-kind, wrong-schema, or mismatched-detector records are
  recomputed.
- `checks/feranchuk_vs_zhai_check.py` is a thin diagnostic over canonical
  anchor/cache helpers; duplicate geometry and Monte-Carlo setup were removed.
- Validation app, figure export, and legacy comparison are cache-only; misses
  direct users to `cxr remote validate`. Remote unavailability never falls back
  to heavy local cache population.
- No equation, validation status, or `signed-off` ledger state changed.
- Independent read-only physics/configuration review found five cache/local-run
  gaps; all were fixed and re-review returned PASS. Seeded legacy-versus-case
  parity produced identical trajectories and zero spectrum difference.
- Fresh local evidence: 180 profile/core, 149 CLI-contract, 79 Zhai/app, and
  326 remote tests passed; lint, typecheck, CLI generated checks, marimo check,
  and validation-app HTML smoke passed. Full suite: 2081 passed, 39 skipped,
  with the same five unrelated skill-mirror, remote-attach, and sandbox
  forkserver failures. Strict Sphinx completed but failed on six pre-existing
  warnings outside touched docs.
