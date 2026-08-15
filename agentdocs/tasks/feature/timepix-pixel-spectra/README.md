# Timepix per-pixel spectra and angular reconstruction

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

These decisions prevent any implementation slice from being Serena
`one-shot` yet.

## Delegation and required skills

- Owner: `lead-task`; the work crosses angular radiation evaluation, numerical
  reconstruction, detector response, factorized results, performance, and
  scientific validation.
- Required skills: `repo-orientation`, `monte-carlo`, `scientific-library`,
  `performance`, `regression-testing`, `physics-review`, and
  `documentation-maintenance`; use `physics-validation` for any new ledgered
  reconstruction claim.
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
