# Python API workflow

Use the Python API for scalar, in-memory simulations and programmatic case construction. Use `pyrite run` for named profiles, resumable checkpoints, remote execution, and the checkpoint-driven analysis apps.

For comparison curves or heatmaps of vacuum line-centre photon energies across
materials, beam energies, and angles, use [`pyrite.photon_energies`](../api.md#photon-energy-comparisons).
It returns arrays without transport; its beam-energy input is in eV.

## Run one scene in memory

The root package exposes the supported high-level objects. `simulate` accepts the scene's three components as separate arguments; it does not accept a `Scene` object directly.

```python
import pyrite as pr

beam = pr.Beam(
    energy_keV=30.0, transverse=pr.TransverseSpot(fwhm_x_mm=1.0), longitudinal=pr.PointBunch()
)
target = pr.Slab("hopg", thickness_ang=10_000.0, tilt_deg=30.0, footprint=pr.Footprint(5.0, 5.0))
detector = pr.Detector()

result = pr.simulate(
    beam,
    target,
    detector,
    physics=pr.Physics(),
    numerics=pr.Numerics(
        cxr_trials=450,
        brem_trials=100,
        cxr_reflections=pr.CxrReflections(n_families=4),
    ),
)
```

`Beam.energy_keV`, target thickness, and target angles must be scalar in a single scene. Put multiple values in a `Sweep` instead. Beam energy is in keV; target thickness is in Å; angles are in degrees.

`simulate` runs the existing typed-case Monte Carlo path and returns a `pyrite.Result`. It does not read or write the workspace, checkpoint cache, or profile catalog.

## Scalar detector acceptance

Specify any two full angular spans or solid angle:

```python
detector = pr.Detector(
    observation_angle_deg=135.0,
    acceptance=pr.AngularAcceptance(polar_deg=2.0, azimuthal_deg=4.0),
)
# Equally valid: polar_deg + solid_angle_sr, or azimuthal_deg + solid_angle_sr.
print(detector.polar_acceptance_deg, detector.azimuthal_acceptance_deg, detector.solid_angle_sr)
```

`AngularAcceptance` describes a patch bounded by constant polar and azimuthal
angles. Its polar interval must stay within 0–180° and its azimuthal span must
be at most 360°. Resolution uses the detector's observation angle; a positive
centred patch at either pole is impossible.

For a flat sensor facing the source, specify its radial distance and full size,
all in millimetres:

```python
detector = pr.Detector(
    observation_angle_deg=135.0,
    acceptance=pr.SensorAcceptance(distance_mm=400.0, width_mm=14.08, height_mm=14.08),
)
```

Height runs along the polar direction and width along the local azimuthal
tangent. The solid angle is exact; the reported spans are local opening angles,
so they do not define a spherical longitude/latitude patch. Acceptance objects
and resolved values are immutable. They describe scalar central-direction
scoring; use `PlanarDetector` for pose, pixels, filters, or face integration.
See [the acceptance derivation](../physics/detectors/detector-solid-angle.md).

`Detector()` keeps the existing material/profile acceptance defaults.
`Detector(polar_acceptance_deg=..., solid_angle_sr=...)` was removed in 0.14.0;
use `Detector(acceptance=AngularAcceptance(...))`. Equal resolved polar spans and solid angles retain the same lowered
case and checkpoint identity. Profile TOML and detector CLI fields are unchanged.

## Describe the beam

`Beam` takes exactly one description per kind of property, so an invalid mix cannot be built. Both `transverse` and `longitudinal` are required:

```python
pr.Beam(
    energy_keV=30.0, transverse=pr.TransverseSpot(1.0), longitudinal=pr.PointBunch()
)  # 1 mm round spot
pr.Beam(
    energy_keV=30.0, transverse=pr.TransverseSpot(0.5, 0.25), longitudinal=pr.PointBunch()
)  # FWHM x/y in mm
pr.Beam(energy_keV=30.0, transverse=pr.PointSpot(), longitudinal=pr.PointBunch())  # point source
pr.Beam(
    energy_keV=30.0, transverse=pr.TransverseSpot(1.0), longitudinal=pr.BunchLength(100.0)
)  # RMS fs
# With a local GPT snapshot: pr.Beam.from_gdf("beam.gdf", z_origin_m=0.0, time_s=1e-9)
```

`transverse` also accepts a Courant–Snyder `TransverseDistribution`, and `longitudinal` accepts `Offsets` or a `LongitudinalDistribution` policy. Divergence is derived: `beam.derived_divergence_mrad(energy_keV)`. Sweep axes address the sub-objects, for example `"beam.transverse": [pr.TransverseSpot(0.5), pr.TransverseSpot(1.0), pr.PointSpot()]`.

Targets likewise require `footprint`: a finite `pr.Footprint(width_mm, height_mm)`, or `None` for a laterally infinite crystal. Leaving out any of these three is an error. Use `pr.PointSpot`; the `pr.Point` alias was removed in 0.14.0.

Coherent emission with physical bunch charge needs offsets the estimator can resolve. On an infinite slab, sampled offsets (a finite `Spot` or `BunchLength`) leave a near-zero cross-electron excess that `cxr_trials` far below the bunch's electron count cannot resolve; bins whose signed estimate goes negative are returned unclipped with a `CoherentSamplingWarning` (nonfinite power still raises `CoherentSamplingError`). Use `PointSpot()` with `PointBunch()`, or a finite footprint with `BunchLength` so offsets are averaged analytically.

## Interpret the result

A `Result` keeps every radiation family separate, each on its own photon-energy grid, next to the inputs and records that produced it. Displaying `result` in a notebook or REPL prints this layout with grid sizes and energy ranges instead of raw arrays:

```text
result
├── cxr.incoherent      SpectralDensity   incoherent PXR/CBS, CXR grid
├── cxr.coherent        SpectralDensity   coherent companion, or None
├── characteristic      SpectralDensity   atomic lines on the CXR grid, or None
├── brem                SpectralDensity   continuum on its own wider grid
├── scene, numerics     the resolved inputs you asked for
├── case                low-level resolved transport input (Case)
├── provenance          Provenance: identity, physics models, diagnostics, environment
└── spatial             SpatialResult per-pixel factors, or None
```

Each `SpectralDensity` pairs `photon_energy_eV` (eV) with `density` and a `normalization`. No family includes another: CXR excludes characteristic lines, and nothing includes brem. With a scalar `Detector`, density is response-free source photons per incident electron per eV per sr (`normalization="source"`). It excludes acceptance, beam current, quantum efficiency, and measured-energy redistribution. With a `PlanarDetector`, the arrays are filter-attenuated, solid-angle-weighted observation averages (`"observation-average"`), still per sr; see the spatial-scoring boundary below.

Three derived views cover most analysis:

- `result.spectral_flux(units="detector")` returns a wide polars table on the union of all native grids: `photon_energy_eV`, `photon_energy_keV`, one column per family, and `total`. Each family is exact at its own nodes, linearly interpolated between them, and zero outside its own grid. `units` selects photons/(s eV) into the accepted solid angle (`"detector"`), photons/(s nA eV) (`"nA"`), photons/(s eV sr) (`"sr"`), or the stored per-electron density (`"electron"`); `pyrite.results.FLUX_UNITS` holds the labels. Beam current is the bunch charge times the repetition rate. The accepted solid angle is `result.accepted_solid_angle_sr()`: the case `domega_sr` for a scalar detector, or the summed pixel solid angle for a planar one. For a scalar detector the density is taken at the central observation direction, so `"detector"` assumes it is uniform over the acceptance.
- `result.to_dataframe()` returns a long polars frame with one row per native node: `component`, `mode`, `photon_energy_eV`, `density`, and `normalization`. It never interpolates or adds totals.
- `result.combined_cxr(mode="incoherent")` returns CXR plus characteristic emission as one `SpectralDensity` on the CXR grid. Pass `include_characteristic=False` to keep only the selected CXR mode.

```python
cxr_energy_eV = result.cxr.incoherent.photon_energy_eV
source_cxr = result.cxr.incoherent.density
continuum_energy_eV = result.brem.photon_energy_eV
source_continuum = result.brem.density

flux = result.spectral_flux(units="detector")  # photons / (s eV), with a total column
native = result.to_dataframe()

digest = result.provenance.identity_digest
backend = result.provenance.backend
device = result.provenance.device
resolved_case = result.case
```

`result.provenance` is a typed, read-only `Provenance` record. Each attribute has a docstring, so editor and notebook hover explain it. It is also a mapping, so `result.provenance["identity_digest"]` and `dict(result.provenance)` keep working. Attributes a run did not record are `None` and absent from the mapping.

| Group | Attributes | Present |
| --- | --- | --- |
| Inputs | `scene`, `numerics` (also `result.scene`, `result.numerics`) | always |
| Identity | `identity_digest` (source content key), `xsgen_tables` | always |
| | `observation_identity_digest`, `observation` | planar detectors |
| | `true_spatial_digest`, `response_digest`, `acquisition_digest` | planar detectors with an `Acquisition` |
| Physics models | `stopping_model`, `characteristic_model`, `attenuation_model`, `bremsstrahlung_model` | always |
| Diagnostics | `kinematic_validity` (central-ray dynamical/extinction screen) | always |
| | `adaptive_sampling` (realized electrons, achieved errors, `stop_reason`, `statistics_limited`) | adaptive `Precision` runs |
| | `line_grid_policy`, `line_grid_resolved` | automatic CXR grids |
| Environment | `backend`, `device`, `git_sha`, `git_dirty`, `versions` | always; git fields are `None` for an installed wheel |

`provenance.resolved_physics` records in-tree descriptor IDs, model-generation
markers, parameters and per-case table digests. Capability declarations describe
the selected model; they do not certify its physical validation domain.

Use `result.scene` for what you asked for. `result.case` is the lower-level resolved transport input: its keys use internal units and historical names (`E0_keV`, `E_grid_line`, `theta_obs_rad`, `domega_sr`) because they feed content identity. Record the digest and software revision with any derived figure or table.

`Physics.emission` accepts a single mode string or a nonempty tuple, list, set,
or frozenset containing `"incoherent"`, `"coherent"`, or both. It stores a
deduplicated immutable tuple in incoherent-then-coherent order, so input order
does not affect hashing or simulation identity. The default is `("incoherent",)`.
Use `Physics(emission=("incoherent", "coherent"))` to select both modes. CLI,
catalog, and checkpoint policies retain their existing tri-state format.

`Physics(emission=("incoherent", "coherent"))` and `Physics(emission="coherent")` expose both CXR modes from the
same transport: `result.cxr.incoherent.density` always means incoherent CXR,
and `result.cxr.coherent.density` is its coherent companion. Neither includes
atomic characteristic emission. Use `result.combined_cxr(mode="coherent")`
for the coherent CXR-plus-characteristic density and its grid. The coherent
model remains inside the [documented validation boundary](../physics/radiation-physics/coherent-emission.md).

`Physics(brem_source=None)` disables continuum scoring and normalizes to
`"none"` for case/checkpoint identity; the string remains accepted. The default
`"mc"` selects simulated continuum.

The high-level API currently cannot inject an external bremsstrahlung array. `brem_source="external"` and `brem_source="none"` therefore return zeros on the resolved continuum grid; use `brem_source="mc"` for simulated background.

## Place a finite filter over detector pixels

A physical detector and filter are downstream photon objects. They do not enter electron transport, so all angular tiles reuse one transported electron population. This example places a silicon plate over part of one Timepix3 chip:

```python
pose = pr.PlanarPose.from_observation(distance_mm=400.0, polar_deg=90.0)
detector = pr.PlanarDetector.timepix3_chip(pose, scorer=pr.PixelScorer(angular_shape=(2, 2)))
filter_pose = pr.PlanarPose.from_observation(
    distance_mm=200.0,
    polar_deg=90.0,
    offset_mm=(3.52, 0.0),
)
half_filter = pr.FilterPlate(
    "silicon",
    thickness_mm=0.1,
    size_mm=(7.04, 14.08),
    pose=filter_pose,
)

pixel_result = pr.simulate(
    beam,
    target,
    detector,
    filters=(half_filter,),
    numerics=pr.Numerics(cxr_trials=450, brem_trials=100),
)
spatial = pixel_result.spatial
assert spatial is not None

energy_eV, selected = spatial.spectra(
    pixels=[(128, 64), (128, 192)],
    component="cxr",
)
line_image = spatial.image((energy_eV[0], energy_eV[-1]), component="cxr")
metadata = spatial.pixel_metadata(pixels=[(128, 64), (128, 192)])
shadow = spatial.transmission_image(5_000.0, component="brem")
```

A pixelated detector always returns `spatial`; its `PlanarDetector.scorer` sets the angular partition and defaults to `PixelScorer()` (one tile). `simulate(pixel_scorer=...)` was removed in 0.14.0; pass the scorer to `PlanarDetector`.

Selected spectra include each pixel's solid angle and therefore have units of photons per incident electron per eV. `component="cxr"` and `"coherent"` are PXR/CBS only; `"characteristic"` is the atomic-relaxation component, and `"cxr_total"`/`"coherent_total"` add it to the corresponding line. `Result.cxr.incoherent.density` remains a detector- averaged density per sr. `SpatialResult` stores tile spectra, pixel rays, and attenuation coefficients as separate factors; `spectra` materializes only the selection and `image` works in bounded pixel chunks rather than allocating a full `(row, column, energy)` cube. Pass `measured=True` to either method to apply a configured `PlanarDetector.response` explicitly.

`pixel_metadata` returns, for a selection only, each pixel's detector-local and lab position, unit direction with lab polar/azimuth angles, distance, scored solid angle, angular tile (and its representative direction when retained), and per-filter path length. `filter_coverage()` marks which centre rays cross each plate, and `transmission_image(energy_eV, component=...)` is the primary transmission at the nearest stored node of that component's energy grid (`energy_node` reports it); an energy outside the grid is an error rather than a snap to its end, so use `"brem"` for the wide continuum grid. Uncovered pixels are exactly 1.

By default each pixel takes the intrinsic spectrum of the angular tile containing it (`PixelScorer(reconstruction="nearest_tile")`), so a smooth angular gradient such as takeoff-angle self-absorption renders as tile-sized steps. `PixelScorer(angular_shape, reconstruction="bilinear_tile")` instead blends the neighbouring tiles' spectra with convex bilinear weights at the same transport cost. Pixels beyond the outermost tile centres keep the edge tile's spectrum, and because blending mixes spectra at fixed energy, a line whose energy shifts with angle (PXR/coherent) broadens or doubles instead of shifting. Use it for smooth components and raise `angular_shape` for angle-dispersed lines. The mode is part of the true-spatial identity. See the [`pixel-angular-interpolation`](../validation/ledger-detector-forward-models.md#pixel-angular-interpolation) ledger row.

Pixels use centre rays from the target reference point. The plate is a finite oriented box, so translation, distance, rotation, oblique thickness, edge misses, and side escape affect the shadow. The primary model includes only Beer--Lambert attenuation: no filter scatter, fluorescence, diffraction, or secondary photons. A `PlanarDetector` without `PixelGrid` supports one centre ray and scalar output; partial-coverage scoring requires a physical `PixelGrid`, whose detector carries the `PixelScorer`.

`provenance["identity_digest"]` remains the source-case digest. Physical observations additionally carry `observation_identity_digest`, which includes ordered geometry, resolved compositions, angular partition, response, and hashes of the coefficient arrays. Plate display names are not identity. An acquisition-enabled observation instead records independent `true_spatial_digest`, `response_digest`, and `acquisition_digest` layers, then links them through `observation_identity_digest`. Changing exposure or reporting edges therefore leaves the intrinsic and true-spatial identities unchanged; changing only the response leaves true-spatial identity unchanged.

```python
counting_detector = pr.PlanarDetector.timepix3_chip(
    pose,
    response=pr.IdealPhotonCounter(),
    scorer=pr.PixelScorer(angular_shape=(2, 2)),
)
acquisition = pr.Acquisition.uniform(
    exposure_s=1.0,
    minimum_eV=0.0,
    maximum_eV=20_000.0,
    bin_width_eV=400.0,
    hit_threshold_eV=500.0,
)
configured = pr.simulate(
    beam,
    target,
    counting_detector,
    acquisition=acquisition,
    numerics=pr.Numerics(cxr_trials=450, brem_trials=100),
)
selected = configured.acquire(pixels=[(0, 0), (10, 12)])
total_image = configured.acquisition_image(pixel_chunk=1024)
window_image = configured.acquisition_image((4_000.0, 8_000.0), pixel_chunk=1024)
```

`IdealPhotonCounter` is explicitly unit-efficiency and energy-preserving; it is a mathematical reference response, not calibrated hardware. `selected.expected` contains registered expected counts per reporting bin; `selected.realized` is present only in Poisson mode, and `selected.counts` selects the configured form. Underflow, overflow, and below-cut accounting are separate arrays. `total_counts` sums registered bins; `window_counts` and the image energy range require bounds that match configured reporting edges so an integer realization is never fractionally split. Images process bounded pixel chunks rather than allocating a full pixel-by-energy cube.

For automatic reporting bins, supply exposure and the native policy only:

```python
native_result = pr.simulate(
    beam,
    target,
    counting_detector,
    acquisition=pr.Acquisition.native(exposure_s=1.0),
    numerics=pr.Numerics(cxr_trials=450, brem_trials=100),
)
native_counts = native_result.acquire(pixels=[(0, 0)])
resolved_edges = native_counts.measured_edges_eV
policy_source = native_result.provenance["reporting_bins"]
```

The request is a `NativeAcquisition`; the result freezes one concrete
`Acquisition` before observation identity or counting. Selection uses the sorted
union of all component response-output edges, or source midpoint-cell edges for
`IdealPhotonCounter`. Complete native bounds include broadened response tails;
the maximum is 65,536 bins, otherwise use explicit edges. Component/pixel
selection later keeps this same axis. Native and explicit requests with equal
resolved values share counts, identity and seeded Poisson replay. Policy/source
metadata is descriptive and excluded from identity. An immutable stored record
keeps its first writer's provenance when equivalent requests share a digest.

`Numerics.energy_bins` samples true photon energies. Timepix `dE_mc` and `dE_out`
discretize the response's input and measured-output axes; acquisition edges
partition that output for reporting. These histograms approximate continuous
recorded energies and do not model an ADC/ToT quantization law. Ordinary source
grid sizing and explicit grid artifacts remain independent of reporting bins.

### Persist, reopen, and rescore an observation

`pyrite.observations.ObservationStore` keeps observations beside, not inside, intrinsic checkpoints; `pyrite run` writes one per case of a counting-observation profile under `pyrite-output/observations/<stem>/`. A true-spatial object stores only factors -- tile map, solid angles, filter paths, representative directions, per-tile spectra, attenuation -- and every response or acquisition over it is a small record, so rescoring never repeats transport:

```python
import tempfile

from pyrite.observations import ObservationStore, observation_from_result

store = ObservationStore("demo", root=tempfile.mkdtemp())
stored_digest = store.put(observation_from_result(configured))
reopened = store.load(stored_digest)
longer = reopened.rescore(
    acquisition=pr.Acquisition.uniform(
        exposure_s=10.0,
        minimum_eV=0.0,
        maximum_eV=20_000.0,
        bin_width_eV=400.0,
        hit_threshold_eV=500.0,
    )
)
store.put(longer)
longer_image = longer.acquisition_image(pixel_chunk=1024)
```

`load` verifies schema, array checksums, and digests and raises `ObservationStoreError` rather than return damaged data. `rescore` retains the concrete reporting edges when only the response changes;
pass a new `Acquisition.native(...)` request to select native edges again.
`rescore` accepts a new `acquisition`, `response`, `rep_rate_hz`, or `bunch_charge_pc`; geometry, filters, and angular sampling are fixed by the stored factors and need a new simulation.

## Apply detector response explicitly

Simulation output remains free of the configured detector response even when the scene's detector has a response model. Apply that response at read time with `Detector.score`:

```python
from pyrite.detectors import Timepix3

timepix = pr.Detector(response=Timepix3(thickness_um=500.0, bias_v=100.0))
detected_cxr = timepix.score(
    result.cxr.incoherent.photon_energy_eV,
    result.cxr.incoherent.density,
    scale=1.0,
)
```

`scale` is explicit. Supply the physically appropriate solid-angle/current conversion for the observable; `Detector.score` does not infer it from a `Result`. Do not reuse a scale or response across instruments without checking the [detector-response model](../physics/detectors/detector-response.md) and [solid-angle convention](../physics/detectors/detector-solid-angle.md).

## Expand a path-addressed sweep

A `Scene` is useful as the scalar base for a Cartesian product. Axis paths can address nested fields and indexed stack layers:

```python
scene = pr.Scene(beam, target, detector)
sweep = pr.Sweep(
    base=scene,
    axes={
        "beam.energy_keV": [30.0, 45.0, 60.0],
        "target.tilt_deg": [15.0, 30.0, 45.0],
    },
)

expanded = sweep.expand()  # ordered (label, Scene) pairs
cases = sweep.cases()  # base.numerics, then axes; typed Monte Carlo Case records
```

Axis insertion order defines Cartesian-product order. Expanded cases receive deterministic one-based seeds in that order. Invalid field names and out-of-range indexes fail when the `Sweep` is constructed.

`Sweep.expand()` and `Sweep.cases()` do not create checkpoints. For a large or resumable sweep, express the ranges in a profile and use `pyrite run`; for direct low-level execution, `pyrite.montecarlo.run_cases` is the supported kernel entry point.

## Separate physical and execution controls

- `Scene` collects beam, target, detector, filters, acquisition, `Physics`, and `Numerics`.
- `Physics` owns emission, background source, and particle-interaction models, including straggling and collision/radiative cutoffs in eV.
- `Numerics` owns electron counts, sampled true-energy grids (`energy_bins`), integration, and execution controls. Electron counts affect the computed content identity; backend, chunk, and transport-core choices select execution policy.
- `Numerics.cxr_reflections` owns result-affecting truncation and quadrature controls such as reflection families and mosaic nodes.
- `Analysis` is currently a presentation-control value object and a bridge for checkpoint-analysis compatibility. It is not accepted by `simulate`.

Use `physics.straggling` or `physics.elastic_model` for model sweep axes,
`numerics.energy_bins.cxr` for sampled grids, and
`numerics.cxr_reflections.n_families` for reflection truncation.
`Sweep.cases(numerics=...)` replaces the base numerical policy before axes apply.
`numerics.<model>`, `numerics.convergence`, and detector-grid paths were removed
in 0.14.0. Use the canonical paths above; unchanged resolved inputs retain
their content identities.

Select the array backend before importing PyRITE, for example:

<!-- verify: skip (illustrative script invocation, not a pyrite/pyrite-dev command) -->
```bash
PYRITE_MC_BACKEND=cpu python my_simulation.py
```

An explicit `Numerics.backend` validates the already-active backend; it does not switch an imported process to another device.

## Choose persistence deliberately

`Result` is an in-memory return type and has no supported save/load method. Resumable campaigns store result mappings in component checkpoints. Current `.pkl` paths contain versioned HDF5 data; older pickle generations remain readable. See [Working with results](working-with-results.md), the [result schema](../repo-design/storage/result-schema.md), and the [supported API reference](../api.md).

No MCPL or other particle-interchange adapter is currently implemented.

## Migrate spectral names

Use `Numerics(cxr_trials=..., brem_trials=..., cxr_chunk=...)` and
`EnergyBins(cxr=..., cxr_by_energy=..., brem=...)`. Dataclass introspection and
`dataclasses.replace` use these canonical field names. Retired constructor and
attribute names were removed in 0.13 after the two-minor deprecation window.

| Retired name | Canonical name |
| --- | --- |
| `Numerics.n_electrons` | `Numerics.cxr_trials` |
| `Numerics.n_electrons_brem` | `Numerics.brem_trials` |
| `Numerics.spec_chunk` | `Numerics.cxr_chunk` |
| `EnergyBins.line`, `.line_by_energy` | `EnergyBins.cxr`, `.cxr_by_energy` |
| `Sweep.line_grid_policy` | `Sweep.cxr_grid_policy` |
| spatial component `line`, `background` | `cxr`, `brem` |

The flat `Result` attributes (`energy_eV`, `spectrum`, `background_energy_eV`,
`background`, `coherent_spectrum`, `characteristic_spectrum`), flat/positional
constructor and `line_total()` were removed after 0.11. Use the grouped
radiation families, keyword-only constructor and `combined_cxr()`.

Through 0.11, `result.to_dataframe()` returned a pandas frame with units,
normalization, case, and provenance in `DataFrame.attrs`. It now returns a polars
frame: units are fixed and documented, `normalization` is a column, and case and
provenance stay on the result. `result.spectral_flux()` replaces hand-written
current/acceptance scaling and cross-grid totals; see
[Interpret the result](#interpret-the-result). Per-pixel `spatial` data is not
flattened.

Runtime controls are `PYRITE_MC_CXR_CHUNK` and `PYRITE_MC_CXR_BUDGET_MB`.
The retired `PYRITE_MC_SPEC_*` fallback spellings were removed in 0.13.

Persisted Case/record keys, HDF5 group names, checkpoint filenames, identity
payload keys, format tags, and RNG stream identifiers retain their historical
spellings. These compatibility contracts keep existing runs reusable and
preserve seeded observations. Atomic characteristic lines and mathematical
line shapes retain their domain terminology.

## Compare photon energies

`pyrite.photon_energies` returns a NumPy array of vacuum CXR line-centre energies
without running transport. The implementation lives in
`pyrite.materials.emission`. Beam kinetic energy and returned photon energy are
in **eV**; sample and detector angles are in degrees. Inputs may be scalars,
lists, or arrays. The default `mode="broadcast"` pairs numerical inputs using
NumPy broadcasting and retains a leading material axis; all-scalar inputs
return `(n_materials,)`.

```python
import numpy as np
import pyrite as pr
from pyrite.materials.crystal import HC_EV_ANG

observation_deg = np.linspace(20.0, 160.0, 141)
energies_eV = pr.photon_energies(
    ["hopg", "mos2", "wse2"],
    beam_energy_eV=30_000.0,
    tilt_deg=30.0,
    tilt_azim_deg=15.0,
    theta_obs_deg=observation_deg,
)
# Each row is one material's comparison curve, shape (3, 141).
wavelengths_nm = HC_EV_ANG / (10.0 * energies_eV)
```

For a heatmap, either shape independent inputs for broadcasting or request a
Cartesian product with `mode="grid"`:

```python
tilts_deg = np.linspace(0.0, 80.0, 81)
energies_eV = pr.photon_energies(
    ["hopg", "mos2"],
    beam_energy_eV=[30_000.0, 100_000.0],
    tilt_deg=tilts_deg,
    theta_obs_deg=observation_deg,
    mode="grid",
)
# Axis order: material, beam energy, tilt, tilt azimuth,
#             detector polar angle, detector azimuth.
hopg_30keV_heatmap = energies_eV[0, 0, :, 0, :, 0]  # (81, 141)

# The same two-angle map with broadcast inputs:
paired_map = pr.photon_energies(
    "hopg",
    30_000.0,
    tilt_deg=tilts_deg[:, None],
    theta_obs_deg=observation_deg[None, :],
)[0]
```

Grid mode accepts scalar or 1-D numerical axes. Scalar axes retain size one,
empty axes retain size zero, and material order and duplicates are preserved.
Both modes accept `phi_obs_deg`; it participates in shape resolution but does
not change this straight-beam vacuum energy because the beam points along lab
`+z`. Sample `tilt_azim_deg` can affect oblique reflections; a reflection
parallel to the surface normal is azimuth-independent.

The default reflection is the lowest catalogue-pinned family parallel to the
declared surface normal, or the first non-extinct harmonic (orders 1–32) of its
primitive reciprocal cut. The unpinned search tests each element's fractional
basis sum with a relative amplitude tolerance of `1e-5` for CIF rounding; it
does not rank intensities. Examples include HOPG/MoS₂ `(002)`, black phosphorus
`(020)`, and GeP `(20-2)`. `hkl=(h, k, l)` overrides all materials;
`hkl={"hopg": (0, 0, 4)}` overrides just that material. Explicit indices give
kinematics even for extinct planes and do not promise emission intensity.
An unresolved cut requires explicit indices. The catalogue surface orientation
is retained when selecting an oblique reflection; the reflection override does
not change the crystal cut. `catalog=` accepts a loaded `MaterialCatalog` or a
complete catalogue path; omission uses the current configured catalogue.

Each reflection denotes its Friedel pair, choosing the member with positive
beam projection. Grazing projections within `1e-12 * |g|` of zero and energies
above the electron kinetic energy return `NaN`. Invalid inputs and incompatible
shapes raise `ValueError`. These are straight-electron, vacuum resonance
energies, not intensity maxima or broadened spectra. Production in-medium
dispersion can shift line centres; mosaicity, transport energy loss, and
detector response require the simulation API.

## Choose adaptive sampling

Adaptive sampling replaces both fixed counts with one shared count chosen from a precision policy:

```python
numerics = pr.Numerics(
    precision=pr.Precision(
        target_rse=0.1,
        min_electrons=400,
        max_electrons=10_000,
        block_electrons=100,
        observables=("cxr", "brem"),
    ),
)
result = pr.simulate(beam, target, detector, numerics=numerics)
sampling = result.provenance["adaptive_sampling"]
sampling["realized_electrons"]
sampling["statistics_limited"]
```

`min_electrons` defaults to one block (`block_electrons`); the other three budget fields are required. The example is an explicit budget, not a default recommendation. Counts must be multiples of the block size. The policy overrides `Numerics.cxr_trials` and `brem_trials`; omitting `precision` retains fixed sampling. The requested policy stays in `Result.case["adaptive_precision"]` and its identity, while realized counts, achieved errors, guards, and stop reason stay in `adaptive_sampling`. A run reaching its maximum without convergence completes with `statistics_limited=True` and a `CxrYieldStatisticsWarning`.

This mode supports incoherent scalar-detector scenes on the per-electron or CUDA core. It rejects coherent emission, physical detectors and observation directions, trajectory capture, GDF beams, grooves, and particle cascades. The two populations share one target and count in this implementation. The [statistical methods](../computation/statistical-methods.md#adaptive-electron-counts) describe the observables, guards, and limits of the error estimate.

## Reuse bodies as targets and filters

`RectangularBody` describes one homogeneous finite solid: an explicit substance
reference, full width/height and thickness in **mm**. `Target` and `Filter`
assign independent scene roles to that description. Target angles and the
catalog crystal cut describe source orientation; a filter's `PlanarPose`
describes its box centre and local axes in lab mm relative to the target
entrance origin. Reusing a body does not place one in-scene object in two roles.

```python
plate = pr.RectangularBody(
    substance=pr.SubstanceRef(kind="crystal", key="silicon"),
    size_mm=(10.0, 10.0),
    thickness_mm=0.1,
)
target = pr.Target(body=plate, tilt_deg=30.0)
window = pr.Filter(body=plate, pose=pr.PlanarPose((0.0, 0.0, 50.0)), name="window")
physical_detector = pr.PlanarDetector(pr.PlanarPose((0.0, 0.0, 100.0)), size_mm=(10.0, 10.0))
scene = pr.Scene(beam, target, physical_detector, filters=(window,))

body_sweep = pr.Sweep(scene, axes={"target.body.thickness_mm": [0.01, 0.1]})
```

Bodies and wrappers are frozen scalar records; sweep axes expand before case
lowering. Filter body and pose axes work similarly, for example
`filters[0].body.thickness_mm` or `filters[0].pose.center_mm`. Existing
`Slab`/`Stack` constructors and their Å-based thickness fields retain their
meaning. `Target.to_slab()` converts mm to Å without rounding and preserves
the established target angle quantization during case construction.
`Scene.target_spec` and `Scene.filter_plates` expose the legacy values for
source, observation and trajectory consumers. `ResolvedObservation` continues
to take fully lowered plates; use `scene.filter_plates` there.

`SubstanceRef` selects the case-sensitive `crystal` or `medium` namespace;
runnable target recipes are not homogeneous substances. Inline `MediumSpec`
compositions are also valid bodies for filters. `Target` requires a crystal
reference and uses the selected target catalog. A key that would redirect
through a recipe to a different crystal is rejected. `lattice_cut="catalog"`
keeps the catalog crystal cut; `lattice_cut=pr.CrystalCut(...)` lowers to
`Slab(cut=...)` (see "Crystal cuts" below). Arbitrary target poses,
layers, grooves and amorphous entrances are outside this adapter's supported
set.
Use the existing target variants for their established specialized behavior.

`Filter` supports only `interaction="primary-attenuation-only"`. Other
requested interactions fail before computation. `Filter.to_plate(catalog=...)`
can resolve an explicit loaded catalog without fallback; it uses an inline
medium when legacy string lookup would select a different namespace or catalog.
Equivalent legacy and adapted inputs retain source cases, observation identities
and numerical behavior. Standard catalog references also retain their legacy
trajectory snapshots; explicit external or disambiguated references snapshot
their resolved inline composition. Catalog tables, stored schema versions and
CLI commands are unchanged.

## Tilt the bunch front and crystal cut

`TransverseSpot` is the Gaussian transverse descriptor; `Spot` remains a
compatibility alias. Compose a front tilt with the existing spot and local
bunch duration:

```python
beam = pr.Beam(
    energy_keV=30.0,
    transverse=pr.TransverseSpot(fwhm_x_mm=0.001, fwhm_y_mm=0.01),
    longitudinal=pr.BunchLength(rms_fs=0.001),
    bunch_front_tilt=pr.BunchFrontTilt(tilt_deg=45.0, azimuth_deg=0.0),
)
target = pr.Slab(
    "hopg", thickness_ang=500.0, tilt_deg=45.0, footprint=pr.Footprint(width_mm=5.0, height_mm=5.0)
)
```

The front is tilted from the beam-normal plane, without rotating the beam.
Matching its tilt and azimuth to the resolved target makes the front **parallel to the
entrance face**: the imposed incoming time gradient cancels the different
vacuum flight times to that face. Positive tilt means later arrival along the
lab transverse direction selected by the azimuth (0 degrees is +x, 90 is +y).
Angles must be finite and strictly between -90 and 90 degrees.

The RMS duration is conditional on transverse position: it remains the local
thickness of the tilted front, while the full envelope can be much longer.
Initially nonzero tilt supports monoenergetic parallel Gaussian/point analytic
sources with Gaussian timing or `PointBunch()`. GDF, Courant–Snyder, energy
spread and other longitudinal policies fail explicitly.

Parallel to the target is a geometric convention, not a guarantee of radiation
phase matching in a chosen direction. The joint position–time Gaussian average
retains the correlation; the old independent suppression factors cannot simply
be multiplied unchanged. See the [independent derivation](../validation/beam-transport/linear-bunch-front-tilt.md).
A wide phase-matched source can have a narrow angular lobe: resolve that lobe
before interpreting central-direction density as detector-integrated flux.
This source model does not establish experimental feasibility of attosecond
local bunch durations.

For the notebook's ideal HOPG (002) geometry (30 keV electrons, target tilt
84.5 degrees, target azimuth zero, observation polar angle 145 degrees),
the transverse phase-matching choice is:

```python
from dataclasses import replace

scene = pr.Scene(
    beam=beam,
    target=pr.Slab("hopg", thickness_ang=500.0, tilt_deg=84.5, footprint=pr.Footprint(5.0, 5.0)),
    detector=pr.Detector(observation_angle_deg=145.0),
)
front = scene.phase_matched_bunch_front_tilt((0, 0, 2), photon_energy_eV=91.2)
matched_beam = replace(scene.beam, bunch_front_tilt=front)
```

This gives an incoming gradient of 135.785 fs/µm. It cancels the transverse
phase mismatch in that central direction for the c-axis reflection; the
parallel-face choice would instead be 84.5 degrees. For a finite Gaussian
spot, the ideal joint intensity factor becomes the residual timing factor
alone: about 0.981 at 91.2 eV with 0.001 fs local RMS duration. Reversing the
front restores strong transverse suppression. This is a form-factor example,
not a prediction of integrated detector gain or a converged transport spectrum.

`Scene.phase_matched_bunch_front_tilt` leaves the scene unchanged and returns a
`BunchFrontTilt`. The reflection is explicit because different reflections
can require different fronts. Omitting `photon_energy_eV` uses the vacuum
zero-scattering line estimate. The scalar detector's central direction is the
default; pass `observation_direction_lab=(nx, ny, nz)` for another ray or a
planar detector. This method currently supports a flat single slab and matches
its mean crystal orientation -- the catalog cut or the slab's `CrystalCut`,
resolved exactly as the simulated case resolves it -- rather than every mosaic
orientation.
Target tilt and azimuth follow the existing nearest-half-degree case
resolution. The helper uses those resolved angles; bunch-front angles retain
full precision. Thus match the resolved entrance-face angles when setting a
parallel front manually.

### Crystal cuts

A flat single `Slab` (or `Target`) accepts `cut=pr.CrystalCut(...)` to orient
the lattice relative to the entrance face. The cut face normal defines the zero
of `tilt_deg`/`tilt_azim_deg`: `tilt_deg=0` is normal incidence on the cut
face, and tilting moves face and lattice together. The cut fixes how the
lattice sits under that face:

```python
cut = pr.CrystalCut(
    surface_hkl=(1, 0, 1),  # or beam_uvw=(u, v, w); omit both for the catalog axis
    roll_deg=17.0,  # right-handed roll about the face normal
    miscut_deg=5.56512,  # continuous tilt off the face normal, 0 <= miscut < 90
    miscut_azim_deg=200.0,  # sample-frame azimuth the lattice tilts toward
)
target = pr.Slab("hopg", 500.0, tilt_deg=84.5, footprint=None, cut=cut)
```

The lattice rotation is `R_miscut @ R_roll @ R_axis`. `surface_hkl` and
`beam_uvw` are mutually exclusive integer axes and are unrelated to the
reflections being scored. `miscut_deg` cuts the face that far off the selected
plane (toward `miscut_azim_deg`), giving a continuous orientation, since integer
surface planes alone rarely give an exact energy match. Target tilts are still
quantized to half a degree; miscut angles are not.

Omitting the cut, or passing `CrystalCut()` or the catalog axis, leaves the
case payload and identity unchanged. An explicit cut adds `azimuth_rad` and
`recip_miscut_rad` keys and a name suffix such as
`cut=(1,0,1) roll=17 miscut=5.56512@200`, so distinct cuts are distinct
cases. The catalog is never modified. Stored catalog line grids belong to the
catalog orientation, so a reoriented cut resolves its line grid automatically
unless you pass an explicit grid. The cut supports only flat single slabs;
grooves, substrates and stacks are rejected. A miscut also rejects targeted
longitudinal timing modes, whose resonance assumes `g` parallel to the face
normal. `Scene.phase_matched_bunch_front_tilt` uses the same orientation as
transport and spectrum evaluation. Solving for the cut that cancels the front
tilt is tracked separately (#509).

Named or inline profile beam tables use the same descriptor:

```toml
[profiles.example.beam.bunch_front_tilt]
tilt_deg = 45.0
azimuth_deg = 0.0
```
