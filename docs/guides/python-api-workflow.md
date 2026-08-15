# Python API workflow

Use the Python API for scalar, in-memory simulations and programmatic case construction. Use `pyrite run` for named profiles, resumable checkpoints, remote execution, and the checkpoint-driven analysis apps.

## Run one scene in memory

The root package exposes the supported high-level objects. `simulate` accepts the scene's three components as separate arguments; it does not accept a `Scene` object directly.

```python
import pyrite as pr

beam = pr.Beam(energy_keV=30.0)
target = pr.Slab("hopg", thickness_ang=10_000.0, tilt_deg=30.0)
detector = pr.Detector()

result = pr.simulate(
    beam,
    target,
    detector,
    numerics=pr.Numerics(
        n_electrons=450,
        n_electrons_brem=100,
        convergence=pr.Convergence(n_families=4),
    ),
)
```

`Beam.energy_keV`, target thickness, and target angles must be scalar in a single scene. Put multiple values in a `Sweep` instead. Beam energy is in keV; target thickness is in Å; angles are in degrees.

`simulate` runs the existing typed-case Monte Carlo path and returns a `pyrite.Result`. It does not read or write the workspace, checkpoint cache, or profile catalog.

## Interpret the result

`Result.energy_eV` and `Result.background_energy_eV` are the photon-energy coordinates for `spectrum` and `background`. With a scalar `Detector`, both arrays are response-free source photon density per incident electron per eV per sr. They exclude acceptance scaling, source current, quantum efficiency, and measured-energy redistribution. With a `PlanarDetector`, the scalar arrays are filter-attenuated, solid-angle-weighted observation averages; see the spatial-scoring boundary below.

```python
line_energy_eV = result.energy_eV
source_line = result.spectrum
continuum_energy_eV = result.background_energy_eV
source_continuum = result.background

digest = result.provenance["identity_digest"]
backend = result.provenance["backend"]
device = result.provenance["device"]
resolved_case = result.case
```

Provenance also contains the resolved scene, numerics, and PyRITE/NumPy versions. Record the digest and software revision with any derived figure or table.

`emission="both"` computes incoherent and coherent spectra from the same transport. `result.spectrum` remains the incoherent array and `result.coherent_spectrum` holds the coherent companion. With `emission="coherent"`, both names select the coherent array. This model remains inside the [documented validation boundary](../physics/radiation-physics/coherent-emission.md).

The high-level API currently cannot inject an external bremsstrahlung array. `brem_source="external"` and `brem_source="none"` therefore return zeros on the resolved continuum grid; use `brem_source="mc"` for simulated background.

## Place a finite filter over detector pixels

A physical detector and filter are downstream photon objects. They do not enter electron transport, so all angular tiles reuse one transported electron population. This example places a silicon plate over part of one Timepix3 chip:

```python
pose = pr.PlanarPose.from_observation(distance_mm=400.0, polar_deg=90.0)
detector = pr.PlanarDetector.timepix3_chip(pose)
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
    pixel_scorer=pr.PixelScorer(angular_shape=(2, 2)),
)
spatial = pixel_result.spatial
assert spatial is not None

energy_eV, selected = spatial.spectra(
    pixels=[(128, 64), (128, 192)],
    component="line",
)
line_image = spatial.image((4_000.0, 6_000.0), component="line")
```

Selected spectra include each pixel's solid angle and therefore have units of photons per incident electron per eV. `component="line"` and `"coherent"` are PXR/CBS only; `"characteristic"` is the atomic-relaxation component, and `"line_total"`/`"coherent_total"` add it to the corresponding line. `Result.spectrum` remains a detector- averaged density per sr. `SpatialResult` stores tile spectra, pixel rays, and attenuation coefficients as separate factors; `spectra` materializes only the selection and `image` works in bounded pixel chunks rather than allocating a full `(row, column, energy)` cube. Pass `measured=True` to either method to apply a configured `PlanarDetector.response` explicitly.

Pixels use centre rays from the target reference point. The plate is a finite oriented box, so translation, distance, rotation, oblique thickness, edge misses, and side escape affect the shadow. The primary model includes only Beer--Lambert attenuation: no filter scatter, fluorescence, diffraction, or secondary photons. A `PlanarDetector` without `PixelGrid` supports one centre ray and scalar output; partial-coverage scoring requires `PixelScorer` and a physical grid.

`provenance["identity_digest"]` remains the source-case digest. Physical observations additionally carry `observation_identity_digest`, which includes ordered geometry, resolved compositions, angular partition, response, and hashes of the coefficient arrays. Plate display names are not identity. An acquisition-enabled observation instead records independent `true_spatial_digest`, `response_digest`, and `acquisition_digest` layers, then links them through `observation_identity_digest`. Changing exposure or reporting edges therefore leaves the intrinsic and true-spatial identities unchanged; changing only the response leaves true-spatial identity unchanged.

```python
counting_detector = pr.PlanarDetector.timepix3_chip(
    pose,
    response=pr.IdealPhotonCounter(),
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
    pixel_scorer=pr.PixelScorer(angular_shape=(2, 2)),
    acquisition=acquisition,
)
selected = configured.acquire(pixels=[(0, 0), (10, 12)])
total_image = configured.acquisition_image(pixel_chunk=1024)
window_image = configured.acquisition_image((4_000.0, 8_000.0), pixel_chunk=1024)
```

`IdealPhotonCounter` is explicitly unit-efficiency and energy-preserving; it is a mathematical reference response, not calibrated hardware. `selected.expected` contains registered expected counts per reporting bin; `selected.realized` is present only in Poisson mode, and `selected.counts` selects the configured form. Underflow, overflow, and below-cut accounting are separate arrays. `total_counts` sums registered bins; `window_counts` and the image energy range require bounds that match configured reporting edges so an integer realization is never fractionally split. Images process bounded pixel chunks rather than allocating a full pixel-by-energy cube.

## Apply detector response explicitly

Simulation output remains free of the configured detector response even when the scene's detector has a response model. Apply that response at read time with `Detector.score`:

```python
from pyrite.detectors import Timepix3

timepix = pr.Detector(response=Timepix3(thickness_um=500.0, bias_v=100.0))
detected_line = timepix.score(
    result.energy_eV,
    result.spectrum,
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
cases = sweep.cases(pr.Numerics())  # typed Monte Carlo Case records
```

Axis insertion order defines Cartesian-product order. Expanded cases receive deterministic one-based seeds in that order. Invalid field names and out-of-range indexes fail when the `Sweep` is constructed.

`Sweep.expand()` and `Sweep.cases()` do not create checkpoints. For a large or resumable sweep, express the ranges in a profile and use `pyrite run`; for direct low-level execution, `pyrite.montecarlo.run_cases` is the supported kernel entry point.

## Separate physical and execution controls

- `Scene` owns physical configuration and model switches: beam, target, detector, emission mode, X-ray dispersion, and background source.
- `Numerics` owns electron counts and execution controls. Electron counts affect the computed content identity; backend, chunk, and transport-core choices select execution policy.
- `Numerics.convergence` owns result-affecting truncation and quadrature controls such as reflection families and mosaic nodes.
- `Analysis` is currently a presentation-control value object and a bridge for checkpoint-analysis compatibility. It is not accepted by `simulate`.

Select the array backend before importing PyRITE, for example:

<!-- verify: skip (illustrative script invocation, not a pyrite/pyrite-dev command) -->
```bash
PYRITE_MC_BACKEND=cpu uv run python my_simulation.py
```

An explicit `Numerics.backend` validates the already-active backend; it does not switch an imported process to another device.

## Choose persistence deliberately

`Result` is an in-memory return type and has no supported save/load method. Resumable campaigns store result mappings in component checkpoints. Current `.pkl` paths contain versioned HDF5 data; older pickle generations remain readable. See [Working with results](working-with-results.md), the [result schema](../repo-design/storage/result-schema.md), and the [supported API reference](../api.md).

No MCPL or other particle-interchange adapter is currently implemented.
