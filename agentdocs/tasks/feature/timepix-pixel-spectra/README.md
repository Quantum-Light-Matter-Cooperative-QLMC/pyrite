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

The landed spatial result is currently in-memory only. It has no persisted
observation artifact, analysis-app detector-image view, or profile/CLI lowering
for physical detector and filter objects. Complete the user workflow: configure
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

## Checklist

1. Define representative Timepix use cases: at least one 256 by 256 chip and
   one explicit 512 by 512 assembly/grid, with detector distance, pose, active
   extent, energy grids, and representative CXR line/background cases. Record
   module-gap limitations rather than inventing a hardware preset.
2. Establish a high-fidelity angular oracle on bounded cases by comparing
   coarse `(1, 1)`, `(3, 3)`, `(5, 5)`, and finer grids against direct
   direction evaluation for selected pixels/regions. Measure line centroid,
   integrated line flux, continuum, and filtered/unfiltered errors separately.
3. Review reconstruction choices before implementation: retain nearest-tile
   inheritance, add conservative bilinear reconstruction on the detector face,
   add a line-aware representation, or use adaptive tile refinement. The
   selected rule must remain nonnegative, define edge behavior, and document
   what quantity is conserved.
4. Replace or extend the bare `angular_shape` only as required by that review.
   Sampling controls remain per `PixelScorer`/physical detector and must be
   canonicalized into `observation_identity_digest` and provenance.
5. Materialize a discrete accepted-flux spectrum for every requested pixel on
   a shared, documented true-energy grid. Preserve exact per-pixel solid angle
   and positioned-filter transmission after angular reconstruction.
6. Apply the configured `Timepix3` response per requested pixel to produce the
   measured-energy spectrum. Prove response-axis/normalization behavior and
   keep response calibration status visible; do not silently claim pixelwise
   threshold/gain nonuniformity that the model does not contain.
7. Keep storage and execution factorized. Selected pixels/regions may
   materialize spectra; images remain chunked; requesting all pixels must be an
   explicit potentially large operation with bounded computation and clear
   memory behavior.
8. Add numerical regressions for angular convergence, azimuthal variation,
   narrow line motion, filter-edge pixels, no-filter compatibility, response
   application, observation identity, and 512 by 512 lazy/chunked behavior.
9. Update the Python workflow and detector-response documentation. Add or
   update validation ledger coverage if the reconstruction introduces a new
   scientific/numerical claim.
10. Specify acquisition/count semantics, add expected-count scaling from beam
    charge/rate and exposure, and add optional seeded integer realizations.
11. Design and implement the separate factorized observation artifact and its
    identity, compatibility, save/load, rescore, and lifecycle behavior.
12. Add analysis-app detector-image, filter-coverage, energy-window, and
    selected-pixel spectrum/histogram views through reusable plotting/data
    logic.
13. Add reviewed profile schema and CLI commands for physical detector,
    response/acquisition/scorer, and ordered filter configuration; preserve or
    deliberately migrate existing scalar detector options.
14. Regenerate the CLI reference and add help, JSON, precedence, validation,
    completion, compatibility, app-loading, artifact, histogram, and
    visualization regressions.

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

Open before implementation:

- Is piecewise-constant tile inheritance sufficiently accurate for expected
  detector extents, or is reconstruction/refinement required?
- If interpolation is required, should it operate on fixed-energy intensity,
  line parameters/centroids, or an adaptive representation? Narrow lines that
  shift with angle make naïve fixed-bin bilinear interpolation suspect.
- Should user control remain `angular_shape`, gain an explicit reconstruction
  enum, or become an error-tolerance/adaptive policy?
- What error tolerance is acceptable for line centroid, integrated line flux,
  and continuum per pixel, and which experimental geometry sets it?
- Does "512 by 512 Timepix" mean a gapless abstract grid or a specific
  multi-chip/module layout whose inactive gaps and chip transforms must be
  represented?
- Is a uniform response matrix across pixels sufficient initially, or must
  measured hardware calibration introduce per-pixel threshold/gain maps?
- Which registered quantity does the requested per-pixel "hit" image show:
  clustered photon events or individual triggered-pixel hits after charge
  sharing?
- Should acquisition threshold be a post-response analysis cut, a configurable
  hardware discriminator input to the Timepix response, or two separately named
  controls?
- Are uniform-width measured bins sufficient, and what explicit lower/upper
  bounds or overflow/underflow bins are required?
- What is the reviewed CLI command tree and migration path from the existing
  scalar `profile set --observation-angle/--polar-acceptance/--solid-angle`
  surface?
- Which observation-artifact location/lifecycle supports both local and remote
  runs without coupling it to intrinsic checkpoint garbage collection?

These decisions prevent any implementation slice from being Serena
`one-shot` yet.

## Delegation and required skills

- Owner: `lead-task`; the work crosses angular radiation evaluation, numerical
  reconstruction, detector response, factorized results, performance, and
  scientific validation.
- Required skills: `repo-orientation`, `monte-carlo`, `scientific-library`,
  `performance`, `regression-testing`, `physics-review`, `cli-ui-ux`,
  `notebook-workflow`, and `documentation-maintenance`; use
  `physics-validation` for any new ledgered reconstruction/counting claim.
- Use `remote-gpu-jobs` and `run-cxr-mc` for representative heavy/final runtime
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
