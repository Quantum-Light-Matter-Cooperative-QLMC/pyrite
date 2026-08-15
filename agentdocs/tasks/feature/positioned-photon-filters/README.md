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

Questions resolved by the design checkpoint below:

- Final public names and whether `PlanarDetector` is nested in `Detector` or is
  itself the scene's physical detector object.
- Exact pose representation: compact detector-local offsets/angles versus a
  reusable validated rigid pose. The public API must make the common
  source--filter--detector line simple while retaining unambiguous 3-D geometry.
- Spatial output contract and the conserved interpolation from coarse angular
  spectra to fine physical pixels.
- Whether the first profile/CLI slice is required for acceptance or follows the
  public Python and result model in a second checkpoint.

## Design checkpoint for supervisor review (2026-08-14)

This checkpoint resolves the open design questions above. Implementation must
not begin until the supervisor reviews this section.

### Public objects and ownership

- Add a bounded `pyrite.instrument` package. Its public frozen value objects
  are `PlanarPose`, `PixelGrid`, `FilterPlate`, `PlanarDetector`, and
  `PixelScorer`; root-lazy exports make all five available from `pyrite`.
  `FilterPlate` is the standalone physical object requested by the user. Do
  not introduce a generic `Filter`, `Volume`, shape protocol, or component
  tree.
- `Scene.filters` is an ordered tuple of `FilterPlate`. `Scene.detector` becomes
  the closed union `Detector | PlanarDetector`: the existing `Detector` remains
  the exact scalar-acceptance compatibility variant, while `PlanarDetector` is
  the physical scene detector. This avoids a second geometry field with a
  duplicate observation angle and avoids hiding physical geometry inside a
  read-time response object.
- `PlanarDetector` owns its plane pose, active size/pixel grid, energy bins, and
  optional response adapter. Geometry is consumed before scoring; its response
  is still applied later through the same `score()` contract as `Detector`.
  Shared scoring plumbing may be extracted, but response models do not acquire
  geometry responsibilities.
- `PixelScorer` is a narrow request object, not a generic actor/tally system. It
  selects the coarse angular grid and requests retention of the factorized
  spatial result. Physical pixel geometry remains on `PlanarDetector`.
- `FilterPlate.material` accepts either a catalog medium/crystal key or the
  already-public `MediumSpec`, whose explicit `(element,
  number_density_A^-3)` composition is the custom-foil path. Do not add a
  formula parser or duplicate a material database in this milestone.
- Optional `name` labels on plates are unique within a scene and retained in
  provenance but excluded from the physical digest, like other display-only
  labels. Tuple order is retained in provenance and the observation digest.

### Frame and pose convention

- Geometry uses one target-centred right-handed lab frame in millimetres:
  origin at the target reference emission point, electron beam along `+z`, and
  the historical scattering plane in `x-z`. Target tilt changes sample-frame
  emission physics; it does not rotate downstream lab objects.
- `PlanarPose(center_mm, normal, x_axis)` is deliberately planar rather than a
  general rigid-transform API. `normal` is local `+z`; `x_axis` is local `+x`;
  local `+y = normal cross x_axis`. Construction normalizes both supplied axes,
  requires finite nonzero vectors and orthogonality within a documented
  tolerance, and stores the normalized tuples canonically.
- `PlanarPose.from_observation(distance_mm, polar_deg, azimuth_deg=0,
  roll_deg=0, offset_mm=(0, 0))` supplies the common source--object--detector
  geometry. At zero roll its local x axis is the increasing-polar direction;
  local y is the increasing-azimuth direction. This is the physical-position
  counterpart of `detector_directions`, not a second angular convention.
- Plate width and height lie along pose-local x/y; thickness lies along local
  z. `PixelGrid.shape` is `(ny, nx)`, pitch is `(pitch_y_mm, pitch_x_mm)`, and
  arrays index `[row_y, column_x]`; row/column zero have negative local y/x
  centres. The initial pixels fill their pitch with no modeled dead rim.
- `PixelGrid.timepix3_chip()` is exactly one 256 by 256 ASIC at 55 micrometre
  pitch (14.08 mm square). It is geometry only and is not the `Timepix3`
  stochastic response. Do not claim the existing, incompletely documented
  multi-chip mounting/gaps as a preset.

### Intersection and attenuation contract

- Rays start at the point-source origin and end at physical pixel centres.
  Transform each unit ray into each plate's local frame and use the standard
  three-axis slab interval intersection against the finite rectangular box.
  Clip the interval to the source--pixel segment. The path length is the exact
  in-box interval length, so side escape near plate edges and oblique
  `thickness / abs(ray dot normal)` are handled by one algorithm.
- Parallel-axis handling is explicit: a ray outside that local slab misses; a
  ray inside leaves that axis unbounded. A scale-aware tolerance handles only
  the parallel/grazing classification. Non-finite inputs, nonpositive
  dimensions/pitches, duplicate names, a source inside a plate, and plates not
  wholly between the source and detector along the central line of sight fail
  during object/scene validation. Overlapping plates are allowed deliberately.
- Pixel coverage is centre-ray coverage. It is exact for the declared point
  pixel scorer, including fully uncovered and covered pixels, but it is not a
  sub-pixel area fraction. Supersampling/finite sensitive-area integration is a
  named future extension.
- Promote a public, CPU `linear_attenuation_inv_mm(material, energy_eV)` in
  `materials.attenuation`; it resolves catalog keys or `MediumSpec`, reuses
  `_mu_total_inv_ang`, validates a one-dimensional positive energy grid, and
  performs the Angstrom-to-millimetre conversion once. Do not expose the
  Monte-Carlo backend or private composition helpers.
- For pixel `p`, angular tile `q(p)`, and plates `j`, the retained primary-beam
  model is

  `F_p(E) = I_q(p)(E) * dOmega_p * exp(-sum_j mu_j(E) * ell_pj)`.

  `I` is intrinsic photons per electron per eV per sr; `dOmega_p` is the exact
  inverse-square/obliquity pixel solid angle; `mu` is per mm; and `ell` is mm.
  Zero plates is an exact multiplicative identity. Plate order cannot affect
  this sum, although provenance preserves it.

### Coarse angular evaluation and conservation

- Generalize the lab-face construction behind `detector_directions` so pixel
  centres, unit directions, and unrescaled pixel solid angles come from one
  coordinate implementation. Map lab directions into the sample frame through
  the existing target-tilt rotation before line/bremsstrahlung evaluation.
- `PixelScorer.angular_shape = (ay, ax)` partitions pixel rows and columns into
  contiguous, near-equal tiles; it need not divide the detector shape exactly.
  Evaluate line and bremsstrahlung spectra once at each tile's solid-angle-
  weighted representative direction, reusing the same transported electrons.
  Default `(1, 1)` retains today's central-direction approximation while still
  resolving every filter/pixel ray.
- Fine pixels inherit their tile's per-steradian spectrum and retain their own
  exact `dOmega_p`. Therefore, with no filters, each tile satisfies the exact
  discrete conservation identity `sum_p F_p(E) = I_q(E) * sum_p dOmega_p`.
  There is no interpolated renormalization and no 65,536-spectrum Timepix run.
- The scalar `Result.spectrum`/`background` on the physical-detector path is the
  solid-angle-weighted detector average, still in the documented per-sr units.
  Applying the detector's total solid angle recovers the sum of unscored pixel
  fluxes. Existing `Detector` scenes bypass this path bit-for-bit.

### Factorized spatial result

- Add optional `Result.spatial: SpatialResult | None`. `SpatialResult` owns one
  shared `PixelRayMap` (`tile_index[ny,nx]`, `solid_angle_sr[ny,nx]`, and
  `path_length_mm[ny,nx,n_filter]`) plus `SpectralFactors` for line,
  bremsstrahlung, and coherent line when present. Each factor stores only
  `energy_eV`, `intrinsic_by_tile[n_tile,n_energy]`, and
  `mu_by_filter_inv_mm[n_filter,n_energy]`.
- `SpatialResult.image(...)` integrates a requested true/measured energy window
  in bounded pixel chunks; `SpatialResult.spectra(pixels=... | region=...)`
  materializes only selected spectra. Neither method constructs an eager full
  `(ny, nx, energy)` cube. Returned per-pixel spectra are accepted flux per
  electron per eV (solid angle included), unlike the scalar per-sr fields.
- Scoring through a configured detector response is likewise chunked and
  explicit. The factorization stores true-energy attenuation before response;
  no response matrix or threshold behavior is mistaken for filter physics.
- A `PlanarDetector` without `PixelGrid` supports only its centre ray and scalar
  output in this milestone. `PixelScorer` and claims about partial coverage
  require a physical `PixelGrid`; construction rejects that unsupported pair.

### Provenance and compatibility identity

- No-filter scenes using the existing scalar `Detector` do not add case keys,
  do not enter the spatial runner, and preserve arrays, `Case` serialization,
  `case_content_key`, checkpoint stems, and component records exactly.
- Downstream objects do not enter the electron-transport `Case` or intrinsic
  checkpoint identity. `Result.provenance["identity_digest"]` remains the
  source-case digest. A physical-detector/filter result additionally records
  `observation_identity_digest`, canonicalized from that source digest, ordered
  physical geometry/scorer values, resolved compositions, and hashes of the
  actual attenuation coefficient arrays. Display labels are excluded.
- Provenance records the ordered plate list, normalized poses, pixel preset or
  explicit grid, angular partition, point-source/centre-ray approximations,
  primary-only interaction model, attenuation-array hashes, and relevant
  PyRITE/NumPy/xraydb versions. Thus changing downstream geometry reuses source
  transport but cannot masquerade as the same observed result.
- Intrinsic checkpoint files remain source artifacts. This milestone does not
  add factorized spatial fields to the version-2 component-store schema. A
  future persisted observation artifact must be keyed by the observation
  digest rather than mixed into the source checkpoint.

### Profile and CLI boundary

- This checkpoint selects a Python-API-first implementation. Filters, poses,
  physical pixel grids, and `PixelScorer` are not flattened into `pyrite
  profile set` options, and this task does not change CLI help/reference.
- Existing profile and checkpoint regressions remain acceptance checks for the
  no-filter compatibility path. Direct profile/TOML serialization and a derived
  observation-artifact lifecycle require a second reviewed checkpoint after the
  object/result contracts have runtime evidence; they are not silently added
  during this implementation.

### Physics record and independent validation plan

- Add `Validation: positioned-filter-attenuation` to the public attenuation/
  application docstring, a ledger row, and
  `docs/validation/detectors/positioned-filter-attenuation.md`. Cite the
  Beer--Lambert law and derive the finite-box path expression above before the
  independent verifier reads its implementation.
- Cheap filters: units (`mu[mm^-1] * ell[mm]`), positivity, zero thickness/
  zero plate identity, normal-incidence path equals thickness, reversal of a
  plate normal leaves path length invariant, grazing/parallel miss behavior,
  and summed optical depth symmetry under plate permutation.
- Independent numeric anchors use direct xraydb/absorption-length values for a
  pure elemental foil and one compound `MediumSpec`; they do not call the new
  public helper to construct expected values. A fresh context writes only the
  validation document and reports the methodology contract. Only a human may
  mark the ledger claim `signed-off`.

### Staged implementation and checks

1. Add public frozen objects, validation, exports, and API tests; amend ADR-0008
   to state that bounded post-emission analytic geometry does not alter the
   closed `Target` set or CUDA navigator.
2. Add shared lab-face/pixel-ray construction and exact vectorized ray--box
   paths with analytic geometry regressions, including translated, moved,
   rotated, edge, side-exit, and overlapping plates.
3. Add the public attenuation helper, equation/ledger/derivation, independent
   Beer--Lambert anchors, and fresh-context verification.
4. Reuse one transport phase for coarse direction-resolved line/bremsstrahlung
   factors; prove tile conservation and `(1,1)` compatibility before enabling
   more angular tiles.
5. Add `SpatialResult` factorization, selected spectrum/image materialization,
   detector-response chunking, observation identity, and the worked partial-
   coverage guide. Keep checkpoints intrinsic.
6. Run focused API/instrument/detector/result tests, existing profile/identity
   and checkpoint compatibility tests, then `pyrite-dev lint`, `typecheck`,
   `docs`, and the appropriate core suite. Any real GPU/runtime confirmation is
   a separate `run-cxr-mc`/`remote-gpu-jobs` step; do not launch a heavy local
   angular sweep.

Remaining implementation risks, not open API decisions: direction-resolved
line cost grows with `ay * ax`; finite pixel sensitive-area integration is not
modeled; and the current point-source reference does not include target emission
extent. These limits are explicit in objects, results, docs, and tests.

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

## Implementation evidence (2026-08-14)

Checkpoint commits:

- `37c2d04` — frozen downstream geometry objects, public ownership, scene
  validation, explicit source-facing pose convention, and ADR boundary;
- `78c6c0a` — shared pixel rays, exact finite-box intersections, moved/rotated/
  side-exit/partial-coverage regressions;
- `1febe00` — homogeneous-material attenuation helper, stacked primary
  transmission, ledger row, and author-prepared verification packet;
- `f1c478b` — one electron transport reused across direction tiles, explicit
  lab-to-sample mapping, `(1,1)` bitwise scalar compatibility, and discrete
  tile-flux conservation;
- `033ace1` — factorized spatial results, bounded materialization and detector
  scoring, physical simulation wiring, separate observation identity, public
  exports, and worked partial-coverage guide.

Acceptance evidence:

- 257 task-focused instrument/material/API/runner/detector/result/checkpoint/
  validation tests pass;
- scoped Ruff and `ty` checks pass; repository-wide `pyrite-dev typecheck`
  passes after syncing declared groups;
- `pyrite-dev docs` and `pyrite-dev repo-map --check` pass;
- the core suite reaches completion with seven unrelated baseline failures in
  catalog/golden/ledger/profile/performance expectations; this branch does not
  change their catalog data, golden files, profile identity, or performance
  owners, and its ledger edit only adds this task's well-formed record/views;
- repository-wide lint remains blocked by the pre-existing undefined `seen` in
  `tests/scan/test_scan_budget.py`.

The `positioned-filter-attenuation` ledger row deliberately remains
`unverified`. Its implementation-side checks are green, but a separate
fresh-context physics validator must audit the prepared packet before status
advances. No profile/CLI or checkpoint-schema surface was added.
