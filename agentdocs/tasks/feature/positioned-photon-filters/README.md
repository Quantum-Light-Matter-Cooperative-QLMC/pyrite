# Positioned photon filters and pixel-resolved detection

Branch: `feature/positioned-photon-filters`

## Problem and scope

Calibration filters are physical objects between the emitting target and the
detector. A scalar transmission factor is insufficient: a finite filter at a
known pose can cover only part of a pixelated detector, and its shadow and
oblique path length depend on filter position, dimensions, orientation, and
source-to-detector geometry.

Add a first-class, material-bearing downstream filter plate and the minimum
planar detector geometry and spatial-scoring surface needed to simulate that
case. Preserve the existing scalar detector API and intrinsic checkpoint path.

This is not a new `Target` variant. `Target = Slab | Stack` owns electron
transport, emission, and self-absorption inside the irradiated material. The
filter is downstream photon geometry owned by `Scene`; detector response remains
a separate digitization step.

Initial physical scope:

- one or more finite rectangular filter plates with material/composition,
  thickness, translation, and orientation in a documented target-centred lab
  frame;
- a planar detector pose and optional rectangular pixel grid, including a
  Timepix-sized preset without conflating chip geometry with the stochastic
  `Timepix3` response model;
- analytic source-to-pixel ray/plate intersections, Beer--Lambert attenuation,
  exact uncovered/covered pixel handling, and oblique thickness correction;
- spatial scoring that can expose a filtered image or selected per-pixel/region
  spectra without requiring an eager full `(ny, nx, energy)` allocation;
- compatibility for scenes with no filters or no pixel grid.

Non-goals for the first implementation:

- general CSG, meshes, arbitrary photon transport, or adding downstream objects
  to the electron-transport CUDA navigator;
- photon scattering, fluorescence, diffraction, or secondary production in the
  filter; the first model is primary-beam attenuation only;
- pretending the current point/reference-source approximation resolves finite
  target emission extent. State and test that approximation, then leave an
  explicit seam for a future phase-space source;
- making OpenMC/GATE-style tally-selection "filters" part of this API.

## External-system evidence and adopted pattern

- Geant4 models absorbers as placed material volumes and attaches sensitive
  detectors or primitive scorers separately. Its segmented examples configure
  detector geometry/cells independently of the hits map. Adopt the separation;
  do not make the physical filter a scoring predicate.
- OpenGATE follows the same split: a volume owns shape, material, translation,
  and rotation; actors attach to volumes, while digitizers form a later chain.
  Its image actors use an independent `size`/`spacing` scoring grid. Adopt a
  material-bearing object plus separate spatial scorer/result contract.
- TOPAS makes every geometry component own parent, type, material, translation,
  and rotation, supplies a `TsPixelatedBox`, and selects a component separately
  from the scoring quantity. Adopt explicit filter and detector geometry with a
  narrow analytic implementation; do not import TOPAS's general component tree.
- SpekPy's ordered `(material, thickness)` filtration is useful only as the
  limiting case: a plate that fully covers the detector at normal incidence
  must reproduce the same energy-dependent Beer--Lambert composition rule.

Primary references:

- <https://github.com/geant4/geant4/blob/master/examples/extended/runAndEvent/RE06/README.md>
- <https://github.com/OpenGATE/opengate/blob/master/docs/source/user_guide/user_guide_volumes.rst>
- <https://github.com/OpenGATE/opengate/blob/master/docs/source/user_guide/user_guide_reference_actors.rst>
- <https://github.com/OpenTOPAS/TOPAS_Official_Documentation/blob/main/parameters/geometry/placement.rst>
- <https://github.com/OpenTOPAS/TOPAS_Official_Documentation/blob/main/examples/Optical/PixelatedDetector.txt>
- <https://github.com/spekpy/spekpy_release/blob/master/SpekPy-Notebook.ipynb>

## Implementation path and likely owners

1. Record the bounded downstream-geometry decision. Clarify ADR-0008: its ban on
   arbitrary target geometry remains; this feature adds only analytic planar
   photon elements after emission. Specify the lab frame, detector-local axes,
   pose convention, overlap/order rules, point-source assumption, and how
   invalid geometry fails at construction.
2. Add frozen public value objects under an owning downstream-instrument module,
   provisionally `FilterPlate`, `PlanarDetector`, and `PixelGrid`. `Scene` owns an
   ordered tuple of filter plates; `Detector.response` continues to own sensor
   digitization. Avoid the bare class name `Filter`, which other transport codes
   use for tally/event selection.
3. Reuse catalog crystal/media compositions and
   `materials.attenuation._mu_total_inv_ang`; promote the smallest stable
   attenuation helper needed by public detector-side code rather than importing
   Monte Carlo internals. Define a documented custom-composition path if catalog
   keys alone cannot represent real calibration foils.
4. Implement vectorized analytic ray intersections from the target reference
   point through detector pixel/tile centres. Accumulate optical depth through
   every intersected plate in scene order; apply `exp(-tau)` before detector
   response. Full coverage at normal incidence must reduce to
   `exp(-mu(E) * thickness)`; no-filter and zero-optical-depth limits must be
   exact identities.
5. Generalize the existing `detector_directions` tiling instead of creating a
   second coordinate convention. Keep angular emission evaluation coarse and
   explicit when necessary, while resolving plate coverage at pixel resolution.
   Define an interpolation/conservation rule so a 256x256 Timepix grid does not
   require 65,536 independent line-spectrum evaluations.
6. Add a spatial result/scorer contract. Preserve current one-dimensional
   `Result` fields and checkpoint identity for the no-filter path. Prefer
   factorized or requested-window output over an unconditional dense spectral
   cube; expose enough provenance to reproduce filter geometry, detector grid,
   angular sampling, and attenuation data.
7. Wire public exports and resolved profile/config serialization only after the
   object and result contracts are stable. Decide whether initial CLI editing is
   repeatable `profile` options or documented TOML; do not flatten pose or pixel
   geometry into unrelated detector scalars.
8. Add the required physics record: source equation, assumptions, units,
   limiting cases, `Validation: positioned-filter-attenuation`, ledger row, and
   fresh-context validation. Document a worked partial-coverage example and the
   unsupported scatter/fluorescence boundary.

Likely owners:

- `src/pyrite/campaign/model.py` and a new bounded downstream-geometry owner;
- `src/pyrite/detectors/spec.py`, `src/pyrite/montecarlo/geometry.py`, and
  `src/pyrite/results/model.py`;
- `src/pyrite/materials/attenuation.py` and catalog composition resolution;
- profile/config lowering only after the Python object model is fixed;
- focused geometry, detector, result, profile, and public-API tests.

## Decisions and open questions

Decided:

- The filter is a standalone physical scene object, not `Target` and not
  `DetectorResponse`.
- Geometry and scoring/digitization remain separate, following
  Geant4/OpenGATE/TOPAS.
- The first shape is a finite rectangular plate with analytic intersections;
  this does not create a general geometry protocol.
- Multiple plates compose by summed optical depth. The first model ignores
  scattering and filter-generated photons.
- Pixel coverage must be spatially resolved; a scalar covered fraction is not
  an acceptable implementation.

Open before implementation:

- Final public names and whether `PlanarDetector` is nested in `Detector` or is
  itself the scene's physical detector object.
- Exact pose representation: compact detector-local offsets/angles versus a
  reusable validated rigid pose. The public API must make the common
  source--filter--detector line simple while retaining unambiguous 3-D geometry.
- Spatial output contract and the conserved interpolation from coarse angular
  spectra to fine physical pixels.
- Whether the first profile/CLI slice is required for acceptance or follows the
  public Python and result model in a second checkpoint.

## Delegation and execution

- Owner: `lead-task`; this crosses public API, bounded geometry, numerical
  representation, physics validation, results, and compatibility boundaries.
- Required skills: `repo-orientation`, `scientific-library`, `physics-review`,
  `physics-validation`, `regression-testing`, and `documentation-maintenance`;
  add `cli-ui-ux` if profile editing/help enters the accepted slice.
- Not Serena `one-shot`: the pose and spatial-result decisions require a
  reviewed design checkpoint before implementation. After that checkpoint,
  analytic intersection/attenuation and focused regression slices may become
  self-contained.

## Acceptance checks

- A finite off-centre plate shadows exactly the intersected detector pixels and
  leaves uncovered pixels bit-for-bit unchanged.
- Moving the same plate along the source--detector axis changes its projected
  coverage consistently with the documented ray geometry.
- Plate rotation changes intersection and oblique path length with correct
  signs, units, and grazing-incidence rejection/tolerance behavior.
- A full-face normal plate matches independent Beer--Lambert reference values
  for an elemental foil and a compound medium across representative energies.
- Two overlapping plates produce `exp(-(tau1 + tau2))`; plate order is
  irrelevant in the attenuation-only limit and retained in provenance for
  future interaction models.
- A no-filter scene reproduces existing scalar spectra, detector scoring, case
  identity, and focused checkpoints without numerical drift.
- Timepix pixel geometry is distinct from `Timepix3` charge sharing/threshold
  response, and the combined path can return a reproducible partially filtered
  pixel image or selected spatial spectra without mandatory dense-cube memory.
- The point-source/reference-surface approximation and unsupported secondary
  filter physics are visible in API documentation and provenance.
- Focused core/API/detector/profile tests, validation checks, lint, typecheck,
  docs, and any changed CLI reference generation pass.
