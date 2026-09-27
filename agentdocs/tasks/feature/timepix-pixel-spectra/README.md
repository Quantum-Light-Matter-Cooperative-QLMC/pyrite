# Pixel-detector acquisition, visualization, and Timepix spectra

Branch: `feature/timepix-pixel-spectra`

## Problem and scope

PyRITE's landed physical-detector path already represents a pixel grid, traces
one centre ray per pixel through positioned material filters, and evaluates a
user-selected coarse angular grid through `PixelScorer.angular_shape`. A
Timepix-sized detector therefore does not require one radiation calculation per
pixel: for example, `(5, 5)` evaluates 25 representative directions and retains
the result in factorized form.

The current angular reconstruction is piecewise constant: every fine pixel
inherits its tile's intrinsic per-steradian spectrum, then receives its own
exact solid angle and Beer--Lambert transmission. This is conservative and
cheap, but pixels in the same tile do not yet acquire distinct spectral shapes
from their distinct polar and azimuthal directions. Establish and implement the
fidelity contract needed for Timepix/Time-over-threshold use, where each pixel
must expose a discrete true-energy and, when requested, measured-energy
spectrum without an eager `(ny, nx, energy)` allocation.

The landed spatial result was originally in-memory only, with no persisted
observation artifact, analysis-app detector-image view, or profile/CLI lowering
for physical detector and filter objects. (As of 2026-09-26 a persisted store,
profile lowering, and partial filter/physical-detector CLI exist; see "Staleness
audit (2026-09-26)".) Complete the user workflow: configure
the detector, acquisition, angular scorer, and filters from the CLI; run or
rescore an observation; persist its factorized products separately from the
intrinsic checkpoint; and inspect detector images and per-pixel spectra in the
maintained analysis app.

Acquisition must support a detector exposure/integration time, total registered
hits per pixel for pixel detectors generally, and Timepix measured-energy
histograms per pixel. Keep physical response resolution, output histogram bin
width (for example 400 eV), and hit threshold (for example 500 eV) as distinct
concepts with explicit units and provenance.

Initial target: arbitrary user-defined rectangular `PixelGrid`, including
512 by 512 assemblies, with a detector-local, user-configurable coarse angular
sampling/reconstruction policy. Preserve one transported electron population
across angular samples and preserve exact per-pixel filter geometry.

Explicitly out of scope:

- CCD intensity-only imaging and spectral recovery through grazing-incidence
  grating dispersion;
- a claim that every physical Timepix assembly is one gapless 512 by 512 plane;
  module layout, gaps, rotations, and calibration require explicit geometry;
- allocating or persisting an unconditional dense pixel-spectral cube;
- filter scatter, fluorescence, diffraction, secondaries, finite-pixel area
  integration, or extended-source phase space;
- treating the existing placeholder Timepix response parameters as calibrated
  hardware truth.

## Reviewed scope expansion (2026-08-15)

### Acquisition and histogram semantics

- Add a frozen acquisition/scoring configuration owned by the physical
  detector observation, not by source transport. At minimum it carries
  positive `exposure_s`, measured-energy histogram edges or positive bin width
  in eV, a nonnegative measured hit threshold in eV, and an optional realization
  seed/mode.
- Expected counts derive from the existing per-incident-electron accepted flux
  and the beam normalization already owned by `BeamSpec`:
  `n_e = exposure_s * rep_rate_hz * bunch_charge_pc * 1e-12 / e`. Exposure,
  threshold, binning, and realization seed must not invalidate or recompute
  intrinsic electron transport.
- Return deterministic expected counts by default. An explicitly requested
  realization draws integer counting statistics with a recorded seed. Never
  present a Poisson draw as the expectation or silently randomize an app view.
- Every pixel detector exposes a two-dimensional total-hit image after its
  configured efficiency/response and threshold. A spectroscopic Timepix
  detector additionally exposes selected per-pixel measured-energy histograms
  and energy-window count images. Preserve true-energy spectra as an inspectable
  pre-response layer.
- Define "hit" before implementation. The current Timepix response matrix
  predicts one clustered recorded photon energy from an incident spectrum; its
  internal charge-sharing Monte Carlo does not retain which neighboring pixels
  fired. The design review must choose and label one of: incident-pixel event
  counts, clustered photon-event counts assigned to an origin/centroid pixel,
  or raw triggered-pixel hit counts. Raw hit maps require a spatial charge-
  sharing operator and cannot be inferred from the current scalar energy
  response matrix.
- Treat measured histogram bin width as output discretization, not detector
  FWHM. A user-provided 400 eV bin width must not overwrite the modeled ToT
  noise/resolution. Likewise, the acquisition hit threshold is distinct from
  the response model's discriminator threshold unless an explicitly documented
  configuration intentionally binds them.

### Persisted observation artifact

- Keep intrinsic checkpoints unchanged and reusable. Add a separate versioned,
  factorized observation artifact keyed by `observation_identity_digest` and
  linked to the source identity digest.
- Persist enough to reopen the result without rerunning transport: detector and
  filter geometry, pixel ray/solid-angle/path factors, coarse intrinsic angular
  factors, attenuation coefficients/hashes, response and acquisition settings,
  beam normalization, and expected/realized-count provenance. Do not persist an
  unconditional dense `(ny, nx, energy)` cube.
- Exposure-only, threshold-only, and histogram-rebin changes should rescore a
  compatible persisted true/response factorization where possible. Define
  exactly which response changes require rebuilding a response matrix but still
  reuse intrinsic angular factors.
- Integrate observation discovery and loading with the maintained analysis
  workflow. Do not mix observation records into the source component-store
  schema or allow two observation digests to masquerade as the same artifact.

### Visualization

- Extend the maintained marimo analysis app with a spatial detector view backed
  by reusable plotting/data functions under `src/pyrite/`; keep app cells thin.
- Required views: expected or realized total-hit image, thresholded/selected
  measured-energy-window image, filter-shadow/coverage diagnostic, and a
  per-pixel true/measured spectrum or Timepix histogram selected by row/column
  or interaction with the image.
- Display exposure, count semantics, threshold, measured bin width/edges,
  detector/filter identity, linear/log scaling, and expected-versus-realized
  status. Pixel hover/selection should expose row, column, local position,
  direction, solid angle, filter path/transmission, and counts.
- Empty/zero-count pixels, missing observation artifacts, unsupported
  non-spectroscopic response, and very large selections must render a useful
  state rather than failing or allocating a dense cube.
- Support the existing analysis launch/export pathway where technically
  compatible; record any limitation if interactive pixel selection cannot be
  preserved in static export.

### CLI and profile configuration

- Physical detector geometry, pixel grid/module layout, response type/settings,
  angular scorer, acquisition settings, and ordered filter plates become
  profile-owned, serializable configuration with effective-value inspection.
- Do not flatten all geometry into unrelated root `profile set` scalars. Design
  reviewed nested detector/filter commands (exact spelling remains open), with
  stable filter identifiers and explicit `show/list/set/add/remove/reset`
  semantics. Existing scalar detector options need a deliberate compatibility
  projection or migration.
- CLI help must state units, defaults, accepted ranges, repeatability/order,
  whether a value changes transport or only observation rescoring, and how a
  hardware preset differs from explicit geometry. Filter commands must preserve
  declared order and make pose/material/thickness/size visible.
- Human output goes to stdout; validation/diagnostics to stderr. JSON-capable
  inspection emits the repository's versioned envelope with normalized units
  and resolved values. Invalid geometry, unknown material, duplicate filter id,
  incompatible detector/response/acquisition combinations, and malformed
  histogram ranges fail at the CLI boundary with exit code 2 and a correction.
- `pyrite run PROFILE` must resolve the profile's physical observation and
  produce/reuse the corresponding observation artifact. Analysis launch must
  resolve the requested profile/material/observation without relying on hidden
  machine-local defaults.

## Existing design and likely owners

Build on, do not duplicate, the completed positioned-filter work:

- `src/pyrite/instrument/model.py`: `PixelGrid`, `PlanarDetector`, and
  `PixelScorer.angular_shape`;
- `src/pyrite/instrument/geometry.py`: exact pixel directions, solid angles,
  and finite-filter path lengths;
- `src/pyrite/api.py`: one transport reused across angular tiles and creation
  of factorized spatial results;
- `src/pyrite/results/model.py`: selected `SpatialResult.spectra(...)` and
  chunked `SpatialResult.image(...)` materialization;
- `src/pyrite/detectors/spec.py` and `detectors/timepix_response.py`: the
  read-time Timepix energy-response adapter;
- completed design/evidence:
  [`agentdocs/tasks/feature/positioned-photon-filters/`](../positioned-photon-filters/),
  the [Python workflow](../../../../docs/guides/python-api-workflow.md), and the
  [filter validation packet](../../../../docs/validation/detectors/positioned-filter-attenuation.md).

Additional owners introduced by the reviewed scope:

- `src/pyrite/campaign/profiles.py`, profile schema/lowering, and
  `campaign/profile_edit.py` for persistent effective configuration;
- `src/pyrite/cli/commands/profile.py` or reviewed nested command owners plus
  generated CLI reference and contract fixtures;
- a separate observation-artifact owner under `results/`, `checkpoints/`, or a
  narrowly named new package chosen by the storage design review;
- `src/pyrite/apps/analysis_ui/` and reusable `plots/` owners for detector
  images and per-pixel spectral inspection.

The smallest owning change should extend `PixelScorer` and the angular-factor
lowering/result contract. Physical pixel geometry and filter attenuation stay
unchanged unless investigation exposes a concrete defect.

## Initial exploration checkpoint (2026-08-15)

Historical: several statements below were superseded; see "Staleness audit
(2026-09-26)" before relying on the call graph or gap list.

This checkpoint is source-level exploration only. It records the seams that
must be stabilized before implementation; no source, test, public-doc, or
runtime changes accompany it.

### Current call graph and constraints

- The maintained run path is scalar:
  `runs.scan._resolved_run` -> `campaign.config.material_sweep` ->
  `campaign.sweep.build_cases` -> `runs.run.run_sweep`. Profile persistence,
  catalog decoding, and `Sweep._normalize_detector` accept only the scalar
  `detectors.Detector`. `PlanarDetector`, `PixelGrid`, `PixelScorer`, response
  acquisition, and ordered filters have no campaign lowering.
- The physical path exists only through `api._simulate_planar`. It builds every
  exact centre ray, partitions them with `angular_tiles`, and calls
  `run_case_directions`. A `(5, 5)` scorer therefore performs one electron
  transport followed by 25 sequential directional spectrum evaluations, not
  25 transports.
- `SpatialResult` already has the correct true-spectrum factorization:
  per-pixel tile index/solid angle/filter paths and per-tile intrinsic spectra
  plus per-filter attenuation coefficients. Fine-pixel intrinsic shapes are
  piecewise constant within each tile; only solid angle and filter transmission
  vary exactly at pixel resolution. Representative tile directions are not
  retained in the result or persisted provenance.
- Scalar checkpoints and the case CAS retain one observation direction and do
  not retain transported electron segments. They cannot later generate a new
  angular grid without transport. A persisted true-spatial factor can rescore
  exposure, threshold, reporting bins, and compatible responses, but detector
  geometry/filter/angular-policy changes require producing a new directional
  factor during a run.
- `Timepix3.score` hides the response's native measured bins by interpolating
  its detected density back onto the true-energy grid. `SpatialResult` also
  requires measured output to preserve that input shape and applies the
  response in a Python loop per pixel. Neither contract can implement native
  400 eV reporting bins or bounded 512 by 512 image scoring efficiently.
- The Timepix response records at most one measured cluster energy for each
  incident photon that triggers any member of a synthetic 3 by 3 neighborhood.
  It sums charge from all fired neighbors, then discards their coordinates and
  multiplicity. The existing matrix therefore models clustered photon events,
  not raw triggered-pixel hits or centroid migration.
- The existing detector tab and detector plotting modules compare aggregate
  one-dimensional response spectra. `AnalysisContext` loads one intrinsic
  checkpoint only; it has no observation inventory or image state. New spatial
  frame/image builders need distinct reusable library owners, with marimo cells
  limited to controls and selection binding.
- A dense 512 by 512 by 1000 float64 cube is about 2.0 GiB before temporaries
  (about 8.4 GiB at 4000 energy samples). The existing factorization is small:
  tile index and solid angle are about 2 MiB each and each filter path map is
  about 2 MiB. Measured images must contract response/window weights in bounded
  pixel chunks; measured histograms materialize only selected pixels/regions.

### Recommended architecture and semantics

1. Keep physical observation configuration beside, not inside, the intrinsic
   `Sweep`. A profile resolver returns the existing scalar sweep plus a frozen
   `ResolvedObservation`-style record containing physical detector geometry,
   response, scorer, ordered filters, and acquisition. The physical detector's
   `scalar_detector()` remains the compatibility projection used to build
   source cases. An absent physical-observation block preserves current profile
   behavior and dataset digests.
2. Add a narrow observation owner with model, identity, and store layers. Do
   not add observation records to checkpoint `{line,brem}` components. Follow
   ADR-0003/ADR-0009: immutable content-addressed HDF5 objects, atomic writes,
   versioned schemas, and a mutable dataset-level index mapping source case
   content keys to observation objects.
3. Layer identity and storage so large true factors are not duplicated:
   `true_spatial_digest` covers source case, geometry, filters, angular policy,
   representative directions, and true arrays; `response_digest` covers the
   response operator/calibration; `acquisition_digest` covers exposure,
   reporting edges, measured cut, realization mode/seed, and normalization.
   The full observation digest links all three plus the source dataset identity.
4. Persist only factorized true data and canonical provenance: pixel tile map,
   solid angles, filter paths, per-tile true spectra, attenuation arrays,
   representative directions/policy, detector/filter configuration, beam
   cadence/charge, approximation flags, schema, and software versions. Pixel
   centres/directions may be deterministically reconstructed from canonical
   geometry, but round-trip validation must prove that hover metadata agrees.
5. Introduce a native measured-response protocol with explicit output edges and
   batch application. Conservatively overlap-rebin native response bins to
   user reporting edges, account explicitly for underflow, overflow, and events
   removed by the measured-energy cut, and add line/background contributions on
   common measured edges. Do not build histograms from the current interpolated
   `score()` output.
6. Label v1 Timepix output **clustered photon events attributed to the
   incident-ray pixel**. One event has one measured cluster energy. Raw neighbor
   triggers, centroid migration, chip-edge loss, pile-up/dead time, and
   per-pixel calibration maps remain explicit non-goals until a spatial
   charge-sharing/calibration model exists.
7. Keep physical discriminator/noise, response input/output discretization,
   modeled FWHM, user reporting-bin edges/width, and downstream measured-energy
   cut as separate named values. A requested 500 eV cut is post-response and
   does not overwrite the response's hardware discriminator.
8. Expected counts are the default. For accepted density per incident electron,
   use `N_e = exposure_s * rep_rate_hz * bunch_charge_pc * 1e-12 /
   scipy.constants.elementary_charge`; zero charge or cadence is the exact
   zero-count limit. An optional realization draws measured-bin Poisson counts,
   and the total image is their sum rather than an independent draw.
9. Make realizations invariant to selection and chunk order with a documented
   coordinate-keyed stream derived from observation digest, user seed, row,
   column, component, and a frozen RNG algorithm/version. Persist aggregate
   realized arrays only if exact replay across future RNG-library changes is a
   required artifact contract.
10. Retain `nearest-tile` as the first explicit reconstruction policy. Do not
    add fixed-energy bilinear interpolation before the oracle: a line moving
    between tile directions can become artificially broadened or double-peaked.
    Evaluate coarse and selected exact directions in one fixed-seed call so all
    comparisons reuse one transported population. Adaptive refinement is the
    preferred next option if nearest-tile misses reviewed tolerances.
11. Add nested profile detector/filter editing only after the domain schema is
    stable. Preserve stable filter identifiers and declared order. Help must
    identify transport reuse/rebuild consequences; effective inspection uses
    normalized units and the versioned JSON envelope. Retain the three scalar
    detector flags as an explicit compatibility projection/migration surface.
12. Extend local production and discovery first, then remote transfer and
    lifecycle. Remote runs must write the observation index/objects and pull
    must hash-validate and atomically install them. Archive, remove, slim, and
    garbage collection need explicit reachability rules so observation objects
    neither disappear while referenced nor accumulate unbounded orphans.

Provisional oracle review targets are centroid error no larger than the lesser
of 40 eV for a 400 eV reporting bin or 0.1 modeled FWHM, integrated line-flux
error of 1--2%, and continuum-band error of 1%. These are design-review
proposals, not validated acceptance criteria.

### Revised staged checklist

1. **Semantics and oracle.** Define the bounded 256 by 256 chip and explicit
   abstract 512 by 512 cases; run fixed-seed direct/coarse direction comparisons
   for `(1, 1)`, `(3, 3)`, `(5, 5)`, and `(9, 9)` at centre, edges, corners,
   polar-only, azimuth-only, and deterministic interior pixels. Separate line
   centroid, line flux, continuum, filter, and Timepix-binned errors. Review the
   tolerance and reconstruction choice before generalizing `PixelScorer`.
2. **Domain and identity schemas.** Freeze physical detector, response,
   acquisition, scorer, and stable ordered-filter records; define scalar
   projection and layered true/response/acquisition identities. Add profile
   decode/resolution without CLI editing and freeze legacy standard-profile and
   checkpoint digests.
3. **Native response and acquisition core.** Expose native measured edges and
   batched response application; implement conservative rebinning, threshold
   accounting, expected-count scaling, coordinate-stable seeded realizations,
   total/window contractions, and explicit generic ideal-counter behavior or a
   reviewed rejection of measured controls for non-spectroscopic detectors.
4. **True-factor store and local runner.** Add the observation HDF5 codec,
   immutable objects, dataset index, corruption/schema checks, atomic partial-
   run recovery, and production alongside intrinsic runs. Prove acquisition
   rescoring reuses true factors while geometry/angular changes do not claim
   reuse that scalar checkpoints cannot provide.
5. **Bounded result APIs.** Extend selected true/measured spectra, pixel
   metadata, total-hit images, energy-window images, filter coverage, and
   histogram selection. Vectorize response/window work per chunk and prove a
   512 by 512 request never allocates a dense pixel-energy cube.
6. **Analysis workflow.** Add lazy observation inventory/selection to
   `AnalysisContext`, reusable spatial frame/plot builders, and thin marimo
   views for expected/realized images, filters, and selected-pixel spectra.
   Define a fixed-selection fallback for static export and useful missing/zero/
   unsupported states.
7. **Profile CLI.** Add reviewed nested detector/response/acquisition/scorer and
   filter `show/list/set/add/remove/reset` surfaces, stable IDs/order, units,
   validation, completion, JSON, dry-run, and scalar compatibility behavior.
   Regenerate CLI contracts and references.
8. **Remote and lifecycle integration.** Transfer and verify observation
   indices/objects with remote runs and pulls; define archive/remove/slim/GC
   reachability; cover partial and corrupt transfers. Finish durable Python,
   detector-response, storage, CLI, and analysis documentation.
9. **Validation and performance closure.** Add focused numerical and storage
   regressions, physics-review the normalization/reconstruction claims, invoke
   fresh-context physics validation for any new ledgered claim, and benchmark
   representative tile counts/chunks on the appropriate backend without a
   local dense 512 by 512 spectral workload.

### Next implementation slices

- Completed: oracle evidence/decision packet, domain/identity schema, the
  native measured-response/acquisition core with minimum bounded count APIs,
  the observation store and producers (item 4), and the profile/CLI surface
  plus `pyrite run` production (item 7); see "Observation store progress" and
  "Sweep production and CLI progress" (both 2026-09-26).
- Next: complete pixel metadata/selection APIs (item 5) and the analysis
  workflow (item 6) over `ObservationStore`; then remote transfer and
  lifecycle (item 8). Workers consume the vertical backend rather than
  inventing parallel representations.

## Decisions and open questions

Decided:

- Coarse angular evaluation is required; one independent Monte Carlo/radiation
  calculation per physical pixel is not the default design.
- A `(5, 5)` request means 25 representative detector-face directions spanning
  both polar and azimuthal variation. It is a starting resolution, not a fixed
  universal default or accuracy claim.
- Pixel positions, solid angles, and filter intersections remain exact at
  pixel-centre resolution even when intrinsic angular spectra are coarse.
- Timepix measured spectra are a downstream read-time response of accepted
  per-pixel true spectra. CCD/grating inference is a later separate task.
- Arbitrary 512 by 512 grids are supported as explicit geometry; no new
  hardware preset is accepted without verified module layout.
- Exposure and histogram settings are downstream acquisition configuration;
  they reuse intrinsic transport.
- Visualization belongs in the maintained analysis app with reusable library
  data/plotting logic, not a new notebook.
- Spatial observations require their own persisted, observation-digest-keyed
  artifact rather than changing intrinsic checkpoint identity.
- CLI configuration is required for physical detectors and filters in this
  milestone; the earlier Python-API-only boundary is retired.
- Human review on 2026-08-23 retained `nearest_tile` as the explicit v1
  reconstruction policy. Every observation records its `angular_shape`; PyRITE
  makes no universal accuracy claim, automatic `(5,5)` default, interpolation,
  auto-sizing, or global pass/fail tolerance. A guaranteed/adaptive policy is a
  separate later design slice.
- v1 Timepix counts are clustered photon events attributed to the incident-ray
  pixel. One accepted photon produces at most one event; charge from triggered
  members of the modeled 3 by 3 neighborhood is summed into one measured
  cluster energy. Raw neighboring-pixel multiplicity, centroid migration,
  chip-edge loss, pile-up, and dead time are not represented.
- Reporting bins are half-open `[edge_i, edge_{i+1})`. Acquisition accepts
  explicit edges or an explicit minimum/maximum/positive-width construction;
  underflow, overflow, and below-cut events remain separate accounting. The
  acquisition cut is post-response and never changes response resolution,
  noise, or discriminator settings.
- A generic pixel detector uses an explicit ideal response: unit efficiency,
  measured energy equal to true energy, and no smearing. This preserves common
  hit-image, threshold, and histogram semantics without presenting it as
  calibrated hardware.
- Timepix v1 uses one uniform response matrix across pixels and labels it an
  uncalibrated model. Per-pixel threshold/gain maps and physical chip/module
  transforms remain future work; arbitrary rectangular grids remain explicit
  abstract geometry rather than hardware presets.
- Deterministic expected counts are the default. Optional realized counts use
  recorded, coordinate-stable seeded Poisson streams and are never presented
  as expectations.

- Observation artifacts live at a separate root,
  `<workspace>/observations/<stem>/{index.json, objects/true/<true_spatial_digest>.h5,
  objects/obs/<observation_digest>.json}`, decoupled from checkpoint
  archive/slim/GC (human decision 2026-09-26). Remote transfer and
  reachability/GC rules for this tree remain item 8.
- Item 4 first landed as a library producer (human decision 2026-09-26); the
  human then delegated the CLI/sweep design with "whatever seems the best
  solution in the long term -- no band-aids". Decided under that delegation:
  - **Authority.** A profile whose physical detector has an acquisition is a
    counting observation. Its sweep's scalar observation angle, polar
    acceptance, and solid angle come from `PlanarDetector.scalar_detector()`,
    so intrinsic checkpoint keys and observation source keys name the same
    transport. Explicit scalar overrides with such a profile are errors; a
    stored scalar detector reference is reported as superseded, never
    silently mixed. Geometry-only physical detectors keep the old scalar
    behavior and digests. The projection drops detector azimuth (existing
    `scalar_detector()` contract); observations use exact geometry.
  - **One transport per case.** Observation directions travel beside the case
    (`run_case`/`run_cases(observation_directions=...)`), never inside it, so
    case content keys, scalar arrays, and CAS blobs are unchanged. All three
    scheduling branches evaluate them on the scalar transport.
  - **Observation line grid.** Directional spectra use a grid resolved jointly
    over every direction (finest sinc spacing, union of feature windows), not
    the scalar grid, whose automatic windows were seeded only along the case
    direction. Evidence: a 6 degree rotation adds seeds the scalar grid lacks.
  - **Zero-energy nodes.** Filters are opaque at an exact 0 eV grid node
    (`mu -> +inf` as `E -> 0+`), with `T = 1` kept for zero path; the standard
    default brem grid starts at 0 eV and previously crashed filtered runs.
  - **CLI tree.** `pyrite profile physical-detector show|set|reset` owns the
    table; `profile filter add|set|rm|list|show` owns plates only. The
    `filter add --detector-*/--shape/--pitch-mm` flags are removed (they
    replaced the whole table and would drop scorer/response/acquisition),
    following the #54/#62 precedent.

Open for their later owning slices:

- Remote runs (`pyrite run --remote`) produce observations on the remote host
  but do not transfer them; item 8 owns transfer, reachability, and GC.
- `api.simulate` keys the source with per-case xsgen tables while sweeps key it
  with the dataset-level table set, so a `produce_observation` result and a
  sweep result for the same physics may be indexed under different source
  keys. Unifying the two marker sets is a separate identity change, tracked in
  [#191](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/191).

The human semantics gate for the domain/native-response/acquisition slice is
closed.

## Domain and identity schema progress (2026-08-23)

Implemented checklist item 2 without adding native count production or CLI
editing:

- frozen `Acquisition` and `ResolvedObservation` records now bind explicit
  physical pixel geometry, response, nearest-tile scorer, ordered filters, and
  half-open measured reporting edges;
- `IdealPhotonCounter` is the explicit generic unit-efficiency,
  energy-preserving response, while Timepix provenance labels clustered photon
  events at the incident-ray pixel and uniform uncalibrated spatial response;
- `pyrite.observation.v2` links independently reusable true-spatial, response,
  and acquisition digests. Filter display labels do not affect identity;
  filter order and resolved composition do. Poisson configuration freezes the
  future coordinate-stream identifier `pyrite.coordinate-philox.v1`;
- profile TOML decodes and resolves nested scorer/response/acquisition records
  through shared campaign lowering. Existing geometry-only profiles continue
  to lower with no implicit response or acquisition, and CLI material lowering
  reuses the same detector/filter builders; and
- acquisition-free simulation retains the legacy monolithic
  `pyrite.observation.v1` provenance and existing exact profile dataset
  digests. Acquisition-enabled simulation emits the layered identity only; it
  does not yet scale exposure, apply native response bins, or produce counts.

Focused observation/profile regressions, the full core suite, lint, typecheck,
repository-map validation, and warning-strict documentation build passed at
that checkpoint.

## Native response and acquisition progress (2026-08-23)

Implemented checklist item 3 and the minimum count contractions required to
exercise it end to end:

- detector responses can emit nonnegative event mass on explicit native
  half-open measured bins. Timepix applies its existing clustered-event matrix
  to batches without interpolating back to the true grid; the ideal counter
  integrates energy-preserving sample cells. Responses without native measured
  semantics fail explicitly;
- acquisition conservatively overlap-rebins native event mass to reporting
  bins and keeps in-range below-cut, reporting underflow, and reporting
  overflow as disjoint channels. A cut crossing a native bin uses the same
  uniform-within-native-bin overlap assumption;
- expected counts use exposure, cadence, bunch charge, and the exact SI
  elementary charge. Cadence/charge now belong to acquisition identity while
  leaving source and true-spatial identity reusable; zero cadence or charge is
  exactly zero counts;
- Poisson streams use `pyrite.coordinate-philox.v1`, keyed by observation
  digest, seed, pixel coordinate, and component. Pixel selection and chunk
  order are invariant. Component draws and measured-bin draws are summed for
  total/window images; no independent total draw is made; and
- `Result.acquire()` returns selected-pixel expected/realized reporting bins
  plus accounting channels. `Result.acquisition_image()` contracts total or
  reporting-edge-aligned windows in bounded pixel chunks. Acquisition-enabled
  simulations retain spatial factors even when the scorer is implicit.

The detector validation ledger now records `pixel-acquisition-counting` as
unverified with implementation-side unit, analytic normalization, conservation,
zero-limit, and RNG invariance checks. This is not independent physics
validation or Timepix hardware sign-off. Focused/neighboring tests (196 passed),
full core (1,860 passed, 74 skipped), lint, typecheck, repo-map/ledger checks,
and warning-strict docs pass. The next slice is checklist item 4: the
factorized observation store/index and local runner production.

## Slice 1 progress (2026-08-21)

Confirmed (interactive review, not silently assumed):

- Reconstruction policy under evidence-gathering: nearest-tile stays v1
  (recommendation #10 accepted as-is). Interpolation/adaptive refinement is
  only designed if the oracle shows nearest-tile insufficient against a bar a
  physics reviewer sets later.
- v1 Timepix hit semantics: clustered photon events attributed to the
  incident-ray pixel (recommendation #6 accepted as-is). Raw triggered-pixel
  hits/centroid migration stay an explicit non-goal pending a spatial
  charge-sharing operator.
- The provisional tolerance numbers (40 eV/0.1 FWHM centroid, 1-2% line flux,
  1% continuum) are evidence-reporting references only, not a pass/fail gate,
  until a physics reviewer signs off a bar from oracle evidence.

Oracle harness: `checks/pixel_reconstruction_oracle.py`. Compares direct
per-pixel evaluation against the landed nearest-tile `SpatialResult`
reconstruction, both drawn from one `run_case_directions` call per
`angular_shape` so the comparison isolates the angular-reconstruction step
from Monte Carlo noise. Covers `(1,1)/(3,3)/(5,5)/(9,9)` at the representative
pixel set (centre, 4 corners, 4 edge midpoints, polar-only, azimuth-only,
interior), and reports line centroid, line flux, continuum, filtered-flux,
and Timepix-binned centroid/flux error per pixel/shape. Not a signed-off
anchor: no `checks/README.md` ledger row, no `Validation:` marker — this is
decision evidence for the open reconstruction-accuracy question, produced at
smoke-test statistics.

Smoke-test run (`N_ELECTRONS=200`, `n_mc=20_000`, one local pass, not
reviewable evidence): max\|line-flux relative error\| fell from 25.2% at
`(1,1)` to 6.0% at `(3,3)`, 3.1% at `(5,5)`, and ~0 (machine precision, as
expected — every fine pixel is its own tile) at `(9,9)`; max\|centroid
error\| fell from 31.8 eV to 5.0 eV over the same range. Full per-pixel
output: `agentdocs/tasks/feature/timepix-pixel-spectra/evidence/oracle-report.json`.
These numbers are noisy at smoke statistics and only span one geometry
(hopg, 30 keV, 60° polar, 9x9 grid, 1.5 mm pitch, 80 mm) — a higher-statistics
`pyrite remote` pass across the representative-geometry set is required before
a physics reviewer can set a real tolerance bar or confirm/reject nearest-tile
sufficiency.

### High-statistics pass (2026-08-21, `qlmc` box)

Ran on the configured remote GPU host (`pyrite remote sync`, then direct SSH
execution of a 20x-electron/10x-Timepix-MC copy of the harness — this is a
standalone `checks/` script, not a scan profile, so it does not go through
the `pyrite run --remote` job/checkpoint queue; see `remote-gpu-jobs`).
`N_ELECTRONS=4000` (was 200), `N_ELECTRONS_BREM=2000` (was 100), Timepix
`n_mc=200_000` (was 20,000); same single geometry (hopg, 30 keV, 60° polar,
9x9 grid, 1.5 mm pitch, 80 mm, half-covering Si filter). Output:
`agentdocs/tasks/feature/timepix-pixel-spectra/evidence/oracle-report.high-stat.json`.

Per-`angular_shape` max / mean over the 12 representative pixels:

| shape | line flux \|err\| | centroid \|err\| | continuum \|err\| | Timepix flux \|err\| |
|---|---|---|---|---|
| (1,1) | 23.8% / 11.9% | 20.2 eV / 11.6 eV | 1.15% / 0.62% | 13.7% / 6.8% |
| (3,3) | 5.9% / 2.9% | 6.6 eV / 3.1 eV | 0.32% / 0.15% | 3.7% / 1.7% |
| (5,5) | 2.95% / 1.9% | 3.1 eV / 2.1 eV | 0.14% / 0.09% | 2.1% / 1.1% |
| (9,9) | 0% / 0% | 0 eV / 0 eV | 0% / 0% | 0% / 0% |

Same monotonic falloff as the smoke run, now on real statistics (Timepix MC
noise floor at 200k draws is well under these effects). Continuum and
centroid land comfortably inside the provisional bar (1% continuum, 40 eV/0.1
FWHM centroid) by `(3,3)` already; at `(5,5)` the worst pixel (`edge_left`,
2.95% line flux) sits just outside the provisional 1-2% band, and it is not
an outlier — `center`, both bottom corners, both horizontal edges, and
`azimuth_only` all cluster at 2.3-2.9%; only pixels whose fine direction is
polar-aligned with its tile center (`corner_tr`, `corner_br`, `edge_right`)
drop near zero.

**Correction (independent physics review, 2026-08-21):** my original read
above — "under-resolved azimuth" — was wrong; see
`evidence/physics-review-slice1.md` for the full independent review. The
residual is entirely **polar**, not azimuthal. The scattering-plane
configuration here (`tilt_azim_deg=0`, in-plane detector) is mirror-symmetric
about the beam/`g`/detector plane, so every scalar entering the line
amplitude (Doppler factor, PXR detuning, escape path) is *even* in the
out-of-plane (azimuthal) offset and *odd/first-order* in the in-plane (polar)
offset — measured directly at +4.5%/deg (line flux) and −7.3 eV/deg (line
peak) vs. ≤1.5% out to the full ±4.3° azimuthal half-span. The "near-zero"
pixels are the ones matching their tile in **polar** angle, regardless of
azimuthal offset. This also means "line flux is the binding metric" was an
artifact of a window-diluted centroid metric (see review §2.2b-c): the
review's recommended per-spectrum-FWHM line-position bar (~0.9 eV for
intrinsic spectra at this geometry) is *more* binding than flux, not less.

Review verdict: **`(5,5)` fails** against a recommended 1% flux / 0.3%
tile-edge-discontinuity / 0.1-intrinsic-FWHM bar — but `(9,9)` (uniform
refinement) is the wrong remedy. The failure is dominated by this oracle's
artificially coarse 1.5 mm pitch (1.074°/pixel, ~27x the real Timepix3
0.039°/pixel); at hardware pitch the same bar admits an ~11x polar
coarsening, so an **anisotropic** grid (roughly `(3, 37)` polar x azimuthal
for a 256-wide chip, not a square tuple) is recommended over both `(5,5)`-style
squares and interpolation. The review also flags harness gaps to close before
this evidence is fully trustworthy: the Timepix branch scores unfiltered
spectra (biases those numbers conservative, not production-representative),
`array_split`'s uneven tiling makes 3 of the 12 representative pixels
polar-exact by construction (biases the reported `(5,5)` mean low, not the
max), and the metric set has no lineshape distance so it would false-pass
naive per-bin interpolation despite splitting the ~9 eV line into two peaks.

This is still one geometry. The review flags the azimuthal null specifically
as a **symmetry** result (holds for any polar angle while beam/`g`/detector
stay coplanar) that **breaks** the moment `tilt_azim_deg != 0` or the
detector sits off-plane — the single most informative next test — plus a
near-normal-polar case and a different target/energy, before trusting an
anisotropic-grid default generally.

Still open before Slice 2 generalizes `PixelScorer`: human sign-off on the
reviewed tolerance bar and the `(5,5)`/anisotropic-grid ruling above (the
review is a recommendation, not a sign-off), the harness fixes it flags, and
the multi-geometry sweep (broken-mirror azimuth first) — plus every other
item in "Decisions and open questions" not listed as confirmed above
(module/hardware-geometry abstraction, generic-detector ideal-response
requirement, measured-bin bounds, CLI command tree, observation-artifact
lifecycle location) — those remain owned by their respective later slices
(2-8) and are not blocking the oracle harness itself.

### Broken-mirror-symmetry pass (2026-08-21, `qlmc` box)

Highest-priority follow-up from the independent review: same scene, but
`Slab(tilt_azim_deg=45.0)` instead of `0.0`, so beam/`g`/detector are no
longer coplanar and the mirror-symmetry argument no longer applies. Same
high-stat settings as the pass above (`N_ELECTRONS=4000`,
`N_ELECTRONS_BREM=2000`, Timepix `n_mc=200_000`), run the same way (remote
sync + direct SSH execution of a disposable, uncommitted harness copy — never
added to `checks/`, per the same evidence-only choice as the prior pass).
Output:
`agentdocs/tasks/feature/timepix-pixel-spectra/evidence/oracle-report.broken-symmetry.json`.

Smoke-stat run first (200/100 electrons, local) already showed the predicted
qualitative shift — `azimuth_only` picked up a real error where before it was
near-zero — confirmed at high stats:

| shape | worst line flux \|err\| | pixel | worst centroid \|err\| | pixel |
|---|---|---|---|---|
| (1,1) | 22.2% | corner_bl | 24.3 eV | corner_br |
| (3,3) | 5.3% | corner_bl | 4.9 eV | corner_br |
| (5,5) | 2.5% | **azimuth_only** | 3.8 eV | interior |
| (9,9) | 0% | — | 0 eV | — |

**The review's prediction is confirmed.** At `(5,5)`, the azimuthal-only
offset pixel now carries the single worst line-flux error (+2.5%), exceeding
both the polar-only pixel (−1.85%) and the recommended 1% bar — where in the
`tilt_azim_deg=0` pass, azimuthal error stayed ≤1.5% even at the full
detector half-span. This was a symmetry result, not a smallness result, and
it breaks exactly as predicted the moment the target azimuth takes the beam/
`g`/detector out of coplanarity.

Consequence for the anisotropic-grid recommendation: it does **not**
generalize as stated. The `~(3, 37)` polar-heavy grid derived from the
`tilt_azim_deg=0` geometry assumed azimuthal error was structurally
second-order; that assumption is geometry-specific, not a property of the
reconstruction method. Any tile-shape default PyRITE ships needs to either
(a) be derived per-geometry from a cheap probe (the review's §5 "auto-sizing
from a 3-direction probe" idea, now looking necessary rather than optional),
or (b) fall back to a symmetric/conservative grid whenever coplanarity can't
be assumed. Still needed per the review's original sequencing: a near-normal
polar-angle case and a different target/energy case, to separate "breaks
because symmetry broke" from "breaks because of this specific 45° choice."

### Near-normal polar-angle pass (2026-08-21, `qlmc` box)

Second review-recommended follow-up: same scene as the original
(`tilt_azim_deg=0`, symmetric, isolating the polar-coefficient-scaling
question from the symmetry-breaking one above), but `polar_deg=20.0` (was
60.0) with a matching `Slab(tilt_deg=20.0)` (was 30.0) — "near-normal
observation with a matching tilt," per the review's §6 item 2. Same high-stat
settings and remote-run procedure as both passes above; harness never
committed. Output:
`agentdocs/tasks/feature/timepix-pixel-spectra/evidence/oracle-report.near-normal.json`.

| shape | worst line flux \|err\| | pixel | mean \|flux err\| | worst centroid \|err\| | mean \|centroid err\| |
|---|---|---|---|---|---|
| (1,1) | 67.3% | corner_br | 40.2% | 558 eV | 355 eV |
| (3,3) | 36.6% | interior | 17.3% | 190 eV | 96 eV |
| (5,5) | 14.1% | corner_tl | 5.8% | 89 eV | 36 eV |
| (9,9) | 0% | — | 0% | 0 eV | 0 eV |

**Surprising result, opposite the direction the review's own formula
suggested.** The review's §6 item 1 cited the polar coefficient
`β sinψ/(1 − β cosψ)` as "→ 0 at near-normal observation, grows toward
grazing," framing near-normal as the likely best case. Measured, it is the
worst case by a wide margin: at `(5,5)`, worst flux error is 14.1% and worst
centroid error is 89 eV here, vs. 2.95%/3.1 eV at `polar_deg=60` and
2.5%/3.8 eV in the broken-symmetry pass — 5-25x larger on every metric, at
every angular_shape, including centroid errors in the hundreds of eV at
`(1,1)`/`(3,3)` where prior passes were tens of eV at most. `(5,5)` is not
close to passing any read of the tolerance bar here; even `(9,9)` would need
checking against a properly windowed centroid metric (this run still uses
the review-flagged diluted metric) before treating it as clean.

Reading the coefficient's *value* going to zero at `ψ→0` is not the same as
its *sensitivity* going to zero — for a relativistic denominator
`(1 − β cosψ)`, the slope of `sinψ/(1 − β cosψ)` at `ψ=0` scales like
`1/(1−β)` (∝ `γ²`), which is large precisely where the coefficient's value is
smallest. That would explain the observed reversal, but this is my own
post-hoc reading of the same formula the review cited, not verified against
the actual `mc_spectrum` implementation the way the review's original
mechanism was — **it needs the same independent-review treatment as the first
finding, not self-adjudication.** Until then: near-normal is not a safe
assumption for a "best case" bound, and the review's `(3, 37)`-style
anisotropic grid (already invalidated by the broken-symmetry pass above) is
now doubly unsafe — both of the review's own generalization boundaries have
broken in the direction of *more* error, not less.

### Oracle harness repair (2026-08-23)

Implemented the evidence-quality repairs required by both independent Slice 1
reviews in `checks/pixel_reconstruction_oracle.py`; this changes no production
physics and establishes no acceptance threshold:

- runs fixed transport/Timepix seeds `(1, 2, 3)` and records the seed on every
  row so the high-variance near-`g` tail can be summarized across realizations;
- reports dominant-peak energy and FWHM, significant-peak count/energies, and
  normalized total-variation lineshape distance in addition to the legacy
  full-window centroid/flux metrics;
- applies each pixel's exact energy-dependent filter transmission before the
  Timepix response, matching the production ordering;
- records tile pixel count, singleton status, and exact pixel-to-tile angular
  offset so `array_split` singleton tiles cannot be read as reconstruction
  success; and
- adds focused pure regressions for the peak/FWHM diagnostic, lineshape metric,
  and filter-before-response ordering.

The committed 2026-08-21 JSON reports retain the old single-seed schema and
remain historical evidence. Do not compare their exact percentages directly
with a repaired-schema run. The repaired harness still needs remote,
high-statistics reruns for baseline, broken-symmetry, detector-on-`g`, and a
near-pole-but-not-`g`-aligned geometry before Slice 1 evidence can be
re-reviewed. Human sign-off on the reconstruction tolerance/default remains
open; Slice 2 must not generalize `PixelScorer` first.

### Repaired multi-geometry pass (2026-08-23, `qlmc` box)

Completed the required four-geometry pass on the configured remote RTX 5080
with CUDA: 4,000 line electrons, 2,000 bremsstrahlung electrons, 200,000
Timepix response samples, and fixed seeds 1/2/3. The harness now exposes named
geometry/statistics/output arguments, so no disposable remote edit was used.
Raw repaired-schema reports and checksums, commands, aggregation rules, tables,
and implementation-context review are recorded in
[`evidence/oracle-report.repaired-summary.md`](evidence/oracle-report.repaired-summary.md).

At `(5,5)`, the median worst intrinsic line-flux error across seeds is 2.95%
for baseline, 2.38% with broken symmetry, 14.1% with the detector on `g`, and
12.8% for the near-pole-but-10°-off-`g` control. Exact filtering amplifies the
detector-on-`g` worst error to about 57.5%; its Timepix-binned flux error is
about 42.7%. The new peak diagnostics expose seed-dependent dominant-feature
switches of 1624→247 eV on `g` and 1519→280 eV in the non-aligned control.

Therefore the evidence-quality rerun is complete, but the design gate is not:
`(5,5)` cannot be accepted as a general default from these data, `(9,9)` is a
singleton identity rather than coarse-reconstruction evidence, and the danger
cannot be confined to exact detector/`g` alignment. Human review must choose
the tolerance and reconstruction/safety policy before Slice 2 generalizes
`PixelScorer`. No production physics or validation-ledger state changed.

## Observation store progress (2026-09-26)

Implemented the library half of checklist item 4:

- `pyrite.observations.ObservationStore` at the decided separate root. True
  objects hold only factorized arrays (tile map, solid angles, filter paths,
  representative tile directions, per-tile intrinsic spectra, attenuation);
  records hold the three canonical layer payloads; `index.json` maps source
  content key to observation digests under an `flock`. Writes are temp +
  `os.replace` in object -> record -> index order; reads verify schema, per-
  dataset SHA-256, payload digests, and the stored attenuation/direction
  hashes. `put` is idempotent and replaces corrupt/truncated objects;
  `verify()` reports broken entries and leftover temporaries.
- `StoredObservation` reopens without transport and reproduces `Result`
  expected and Poisson counts exactly; `rescore(acquisition=, response=,
  rep_rate_hz=, bunch_charge_pc=)` reuses the true payload, and its digest
  equals a fresh `simulate` with the same settings.
- `runs.observe.produce_observation(scene, numerics, store)` reuses a stored
  observation when the true-spatial identity recomputed before transport
  (source key, geometry, ordered filters, scorer, tile directions,
  attenuation on the stored grids) matches, and otherwise simulates and
  stores. Tests prove that acquisition/normalization/response/filter-label
  changes skip transport, while angular-shape, filter-thickness, no-filter,
  and source changes transport.
- Supporting refactors: `observation_identity` split into
  `true_spatial_payload`/`response_payload`/`acquisition_payload`/
  `link_identity` with a frozen-digest regression; `SpatialResult` retains
  `tile_directions_lab`; `instrument.attenuation.attenuation_matrix` replaces
  the private `api._attenuation_matrix` (oracle harness and filter validation
  packet updated); `api.source_identity_digest` exposed the pre-transport key (later replaced by `api.observation_plan`).

Not done in this slice: remote transfer and GC/reachability (item 8).

## Sweep production and CLI progress (2026-09-26)

- `observations.plan`: `PixelSampling` and `ObservationPlan` are the single
  path for sampling, factor assembly, reuse matching, and layered identity,
  used by `api.simulate`, `runs.observe.produce_observation`, and sweeps
  (`SweepObservation`, sampling shared across cases).
- `run_sweep(observation=...)`: reuses stored factors for read-time-only
  changes; evaluates missing observations on the scalar transport; reruns a
  cached case only to fill its observation, leaving the record and shards
  untouched; directional arrays never enter records or CAS blobs.
- `pyrite run` builds the `SweepObservation` from the profile and stores it at
  the sibling `observations/<stem>` of the checkpoint root.
- CLI and docs: `profile physical-detector`, `profile filter set`, `material
  simulate` honoring scorer/acquisition, regenerated CLI reference, rewritten
  sweep-profiles section, Python guide persistence example (executed by the
  doc-block test).
- Real-CLI smoke (16 by 16, 7 electrons, filter over a 0 eV brem grid):
  produce, exposure 1 -> 3 rescored with no transport (totals exactly 3x,
  shared true object), angular-shape change re-transported the observation
  only with `line.h5` untouched.
- Physics review needed (not self-adjudicated): the observation line-grid
  union and the zero-energy opaque branch of `positioned-filter-attenuation`
  (ledger row notes it postdates the 2026-09-13 re-derivation).

## Staleness audit (2026-09-26)

Checked against the branch after rebase onto `main`:

- Stale: "profile persistence ... accept only the scalar Detector" and "no
  campaign lowering" for physical detectors/filters. `materials` catalog now
  decodes `profile_physical_detectors`/`profile_filters`;
  `campaign/observation.py` lowers filters, physical detectors, responses,
  scorers, and acquisitions (`resolve_profile_observation`).
- Stale: "The physical path exists only through `api._simulate_planar`".
  `pyrite material simulate MATERIAL --profile P` runs one physical-detector
  scene through `simulate` (filesystem-free except `--output-file` `.npz`). It
  still passes a default `PixelScorer()` and no acquisition, so it does not
  consume the profile's resolved scorer/response/acquisition. Item 7 owns
  that gap.
- Stale: CLI "no profile/CLI lowering". `pyrite profile filter
  add/list/rm/show` exists, and `filter add --detector-distance-mm/--shape/
  --pitch-mm/--detector-*` creates the profile physical detector. Missing:
  filter `set/reset`, and response/scorer/acquisition editing. `pyrite
  detector` manages scalar named detectors only.
- Resolved by item 3: the `Timepix3.score` interpolation/shape-preserving
  limitation no longer blocks native reporting bins (`native_score`, including
  nonuniform true grids after the #100/#114 rebase).
- Still accurate: `run_case_directions` performs one transport, then loops
  directions; `SpatialResult.spectra(measured=True)` loops per pixel;
  `AnalysisContext` loads one intrinsic checkpoint and has no observation
  inventory.
- Superseded: Slice 1 text saying "Slice 2 must not generalize `PixelScorer`"
  and "human review must choose the tolerance" is closed by the 2026-08-23
  `nearest_tile` decision under "Decisions and open questions".
- Fixed: the skill reference `run-cxr-mc` is now `run-pyrite`; the former does
  not exist.

## Delegation and required skills

- Owner: `lead-task`; the work crosses angular radiation evaluation, numerical
  reconstruction, detector response, factorized results, performance, and
  scientific validation.
- Required skills: `repo-orientation`, `monte-carlo`, `scientific-library`,
  `performance`, `regression-testing`, `physics-review`, `cli-ui-ux`,
  `notebook-workflow`, and `documentation-maintenance`; use
  `physics-validation` for any new ledgered reconstruction/counting claim.
- Use `remote-gpu-jobs` and `run-pyrite` for representative heavy/final runtime
  evidence. Do not benchmark a 512 by 512 full spectral cube locally.
- No delegation slice is self-contained enough for one-shot execution until
  the reconstruction and hardware-geometry decisions are reviewed.

## Acceptance checks

- A user can configure an explicit detector pixel grid and a coarse angular
  policy such as 5 by 5 without evaluating one intrinsic spectrum per pixel.
- Selected pixels at different polar and azimuthal directions return discrete
  true spectra with documented units and, with the Timepix response enabled,
  discrete measured spectra on a documented energy axis.
- The chosen angular reconstruction meets reviewed per-pixel centroid, line
  flux, and continuum tolerances against direct selected-pixel/fine-grid
  evaluation for representative geometries.
- Per-pixel solid angle and positioned-filter path/transmission remain exact;
  uncovered pixels and the no-filter path retain their identity limits.
- Angular sampling/reconstruction and response configuration alter the
  observation identity and are fully recorded in provenance.
- A 512 by 512 geometry can produce selected spectra and energy-window images
  with bounded memory. No mandatory `(512, 512, n_energy)` allocation or
  262,144 independent radiation evaluations occurs.
- Existing scalar-detector and positioned-filter focused tests remain green;
  new focused tests, lint, typecheck, docs, and relevant core checks pass.
- Profile/CLI users can create, inspect, edit, order, and remove physical
  detector/filter/acquisition configuration with explicit units and stable JSON
  output; effective configuration round-trips without loss.
- Exposure, beam charge/rate, response efficiency, threshold, and measured
  energy bins produce dimensionally correct expected counts. Seeded realized
  counts are reproducible and clearly distinguished from expectations.
- A completed observation can be reopened after process exit and viewed in the
  analysis app without rerunning transport or loading a dense spectral cube.
- The analysis app renders total-hit and energy-window detector images,
  filter coverage, and selected-pixel true/measured spectra; Timepix views also
  render the configured spectrally binned histogram.
- Timepix outputs explicitly identify whether counts are clustered photon
  events or raw triggered-pixel hits, and tests cover the selected charge-
  sharing semantics.
