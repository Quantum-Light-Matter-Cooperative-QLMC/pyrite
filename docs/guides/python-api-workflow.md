# Python API workflow

Use Python to construct a scene, run a simulation in memory, and inspect its
spectra. For named profiles, resumable checkpoints, or distributed runs, use
[`pyrite run`](running-on-a-cluster.md) instead. The complete object and field
reference is in the [supported Python API](../api.md).

## Choose physical-input imports

| What you are describing | Import from |
| --- | --- |
| Beam energy and analytic or GDF source distributions | `pyrite.beams` |
| Slabs, layered stacks, crystal cuts, footprints, bodies, and target/filter roles | `pyrite.geometry` |
| Catalog materials and their properties | `pyrite.materials` |
| Detector geometry and response | `pyrite.detectors` |
| Scene composition and sweep axes | `pyrite` (`Scene`, `Sweep`) |

The [namespace policy](../development/api-policy.md#canonical-domain-facades-first-migration-slice)
lists every beam and geometry export. `Spot` remains the supported alias of
`TransverseSpot`. Domain imports resolve to the same objects as their historical
root and implementation imports, with the same units and defaults.

## Run one scene in memory

Build a beam, target, and detector with explicit units. `simulate` accepts
these components or a configured `Scene`.

```python
import pyrite as pr
from pyrite.beams import Beam, PointBunch, TransverseSpot
from pyrite.config import CxrReflections, FixedSampling
from pyrite.geometry import Footprint, Slab

beam = Beam(
    energy_keV=30.0,
    transverse=TransverseSpot(fwhm_x_mm=1.0),
    longitudinal=PointBunch(),
)
target = Slab(
    "hopg",
    thickness_ang=10_000.0,
    tilt_deg=30.0,
    footprint=Footprint(5.0, 5.0),
)
detector = pr.Detector()

result = pr.simulate(
    beam,
    target,
    detector,
    physics=pr.Physics(),
    numerics=pr.Numerics(
        sampling=FixedSampling(cxr_electrons=450, brem_electrons=100),
        cxr_reflections=CxrReflections(n_families=4),
    ),
)
```

Beam energy is in **keV**, target thickness in **Å**, lateral dimensions in
**mm**, and scene angles in **degrees**. `Beam` needs both transverse and
longitudinal descriptors; `Slab` needs an explicit footprint (`None` means
laterally infinite). `simulate` returns a `Result` without creating checkpoints.

For reusable numerical/physics settings, load a named preset with
`pyrite.config.load_numerics("default")` or
`pyrite.config.load_physics("electron-standard")`.

## Interpret the result

`Result` keeps radiation components on their own energy grids:
`result.cxr.incoherent`, optional `result.cxr.coherent`,
`result.characteristic`, `result.brem`, and optional channeling or transition
radiation. Each `SpectralDensity` has `photon_energy_eV` and `density`.
Scalar-detector source density is **photons/(incident electron · eV · sr)**,
without detector response or acceptance weighting.

```python
cxr = result.cxr.incoherent
continuum = result.brem
flux = result.spectral_flux(units="detector")  # photons/(s eV), accepted solid angle
native = result.to_dataframe()  # Polars, one row per native energy node
combined = result.combined_cxr()  # CXR + applicable line components; no brem

digest = result.provenance.identity_digest
backend = result.provenance.backend
device = result.provenance.device
resolved_case = result.case
```

`combined_cxr()` adds the applicable line components once; bremsstrahlung
remains separate because its energy grid can differ. `spectral_flux` combines
grids with interpolation and scales using beam current and acceptance;
`to_dataframe` preserves native nodes without interpolation. Save the identity
digest and software revision with derived plots. For checkpoint analysis and
persistence, see [Working with results](working-with-results.md).

## Apply detector response explicitly

A scalar `Detector` does not automatically fold its response into the
returned source spectrum. Score a detector response when you need it:

```python
from pyrite.detectors import Timepix3

timepix = pr.Detector(response=Timepix3(thickness_um=500.0, bias_v=100.0))
detected_cxr = timepix.score(
    result.cxr.incoherent.photon_energy_eV,
    result.cxr.incoherent.density,
    scale=1.0,
)
```

`scale` is an explicit conversion factor: choose the appropriate incident
flux, solid angle, and exposure for the quantity you want. See
[Detector response](../physics/detectors/detector-response.md).

## Expand a path-addressed sweep

Use a `Sweep` for a Cartesian product of scalar scene parameters:

```python
scene = pr.Scene(beam, target, detector)
sweep = pr.Sweep(
    base=scene,
    axes={
        "beam.energy_keV": [30.0, 45.0, 60.0],
        "target.tilt_deg": [15.0, 30.0, 45.0],
    },
)
expanded = sweep.expand()  # nine (label, Scene) pairs
cases = sweep.cases()  # nine resolved Monte Carlo cases
```

These methods *construct* cases; they do not run or persist a sweep. For large
studies use [Sweep profiles](sweep-profiles.md) and
[Running on a cluster](running-on-a-cluster.md).

## Compare photon energies

Compute vacuum CXR line-centre energies without transport. Unlike
`Beam.energy_keV`, this function accepts electron kinetic energy in **eV**.

```python
import numpy as np

angles = np.linspace(20.0, 160.0, 141)
energies_eV = pr.photon_energies(
    ["hopg", "mos2"],
    beam_energy_eV=30_000.0,
    tilt_deg=30.0,
    theta_obs_deg=angles,
)
```

The default broadcast mode retains a leading material axis. Use `mode="grid"`
for a Cartesian product of one-dimensional numerical axes, or pass `hkl=`
for an explicit reflection. This is a kinematic line position, **not** a
prediction of line strength, width, or detected spectrum.

## Separate physical and execution controls

`Physics` chooses physical models; `Numerics` specifies sampling,
discretization, reflection truncation, and local computation. Typed groups
make the distinction explicit:

```python
from pyrite.config import (
    CxrReflections,
    ExecutionConfig,
    FixedSampling,
    RadiationNumerics,
    TransportNumerics,
)

numerics = pr.Numerics(
    sampling=FixedSampling(cxr_electrons=400, brem_electrons=200),
    transport=TransportNumerics(energy_model="midpoint", max_dE_frac=0.01),
    radiation=RadiationNumerics(cxr_reflections=CxrReflections(n_families=6)),
    execution=ExecutionConfig(transport_core="per-electron", cxr_chunk=256, brem_chunk=256),
)
print(numerics.describe_numerics())
```

`TransportNumerics` and `RadiationNumerics` concern numerical accuracy;
`ExecutionConfig` controls local kernels and batches, **not** a cluster
scheduler. Named local/SLURM/Dask executor profiles are separate: see
[Running on a cluster](running-on-a-cluster.md). The flat `Numerics` fields
remain usable; don't specify conflicting values in flat and grouped fields.
The `with_transport()`, `with_radiation()`, and `with_execution()` helpers
update a section of an existing immutable configuration.

## Choose adaptive sampling

Use fixed electron budgets for reproducible fixed-count runs, or request
adaptive stopping on supported incoherent scalar-detector cases:

```python
from pyrite.config import AdaptiveSampling

adaptive = pr.Numerics(
    sampling=AdaptiveSampling(
        target_rse=0.05,
        min_electrons=200,
        max_electrons=20_000,
        block_electrons=100,
        observables=("cxr", "brem"),
    ),
)
# After simulating with `adaptive`, inspect result.provenance.adaptive_sampling.
```

Adaptive limits are budgets, not promised sample counts; check
`statistics_limited` and the realized counts in provenance. Coherent emission,
pixel detectors, and several advanced transport modes are not eligible.
`pyrite.config.Precision(...)` remains an equivalent adaptive policy. See
[Adaptive electron counts](../computation/statistical-methods.md#adaptive-electron-counts)
for estimator limitations.

## Scalar detector acceptance

Use angular spans for a central-direction detector, or physical size and
distance for a flat sensor:

```python
from pyrite.detectors import AngularAcceptance, SensorAcceptance

angular = pr.Detector(
    observation_angle_deg=135.0,
    acceptance=AngularAcceptance(polar_deg=2.0, azimuthal_deg=4.0),
)
finite = pr.Detector(
    observation_angle_deg=135.0,
    acceptance=SensorAcceptance(distance_mm=400.0, width_mm=14.08, height_mm=14.08),
)
```

Scalar scoring uses the central direction; it does not integrate a rapidly
varying radiation pattern over the acceptance. For that use a physical
`PlanarDetector`. See [Detector solid angle](../physics/detectors/detector-solid-angle.md).

## Place a finite filter over detector pixels

A `PlanarDetector` scores angular tiles and can project downstream
attenuation across a pixel grid:

```python
from pyrite.config import FixedSampling
from pyrite.detectors import FilterPlate, PlanarDetector, PlanarPose
from pyrite.observations import Observation, PixelScorer

pose = PlanarPose.from_observation(distance_mm=400.0, polar_deg=90.0)
pixels = PlanarDetector.timepix3_chip(
    pose,
)
plate = FilterPlate(
    "silicon",
    thickness_mm=0.1,
    size_mm=(7.04, 14.08),
    pose=PlanarPose.from_observation(distance_mm=200.0, polar_deg=90.0),
)
pixel_result = pr.simulate(
    beam,
    target,
    pixels,
    filters=(plate,),
    numerics=pr.Numerics(sampling=FixedSampling(100, 50)),
    observation=Observation(detector="detector", sampling=PixelScorer((2, 2))),
)
image = pixel_result.spatial.image((1_000.0, 10_000.0), component="cxr")
```

`spatial.spectra(pixels=[...])` reads selected pixels without materializing
the full pixel × energy cube. Filters use primary Beer–Lambert attenuation;
they do not model scattered or fluorescent secondary photons. Pixel-centre
rays and finite angular tiles limit spatial/angular accuracy. To model
registered photon counts, configure `Observation.response` and an
`Observation.acquisition`, then use `result.acquire(...)` or
`result.acquisition_image(...)`. See the
[detector model](../physics/detectors/detector-response.md) and
[Working with results](working-with-results.md) for stored observations.

## Compose independent observations

Keep the detector's geometry in a scene and put its readout and numerical
scoring choices in an `Observation`:

```python
from dataclasses import replace

from pyrite.detectors import IdealPhotonCounter, PixelGrid, PlanarDetector, PlanarPose
from pyrite.observations import Acquisition, Observation, PixelScorer

panel = PlanarDetector(
    PlanarPose.from_observation(100.0, 60.0),
    pixels=PixelGrid((16, 16), (0.055, 0.055)),
)
scene = pr.Scene(beam, target, panel)
observation = Observation(
    detector="detector",
    response=IdealPhotonCounter(),
    acquisition=Acquisition.native(exposure_s=1.0),
    sampling=PixelScorer((4, 4)),
    name="counts",
)
controls = pr.Numerics(sampling=FixedSampling(100, 50))
counts = pr.simulate(scene, numerics=controls, observation=observation)

longer_exposure = replace(observation, acquisition=Acquisition.native(exposure_s=2.0))
more_counts = pr.simulate(scene, numerics=controls, observation=longer_exposure)
```

Both calls use the same unchanged scene. `detector="detector"` resolves the
registered geometry; a planar value is also accepted when its geometry matches.
For a virtual angular measurement, use `Observation(detector=Detector(...))`.
Observation names do not affect digests. Response or acquisition changes retain
compatible source and true-spatial identities, while changed angular sampling
invalidates affected spatial factors. `simulate` performs transport on each
call; persisted observation production can reuse compatible true-spatial factors.
`Sweep.cases(numerics=controls, physics=models, observation=observation)` binds
the same request to every expanded scene without executing transport. Common
run overrides apply before sweep axes; detector references follow each point's
physical geometry. `quantum=` uses the same inherit/override/disable rules.

Legacy Scene run settings and detector response/scorer constructor arguments
remain supported through compatibility adapters. Observation omission inherits
those values; differing explicit settings raise before transport. Starting with
geometry-only detector construction, as above, allows independent requests
without those conflicts.

### Named CLI pixel profiles use the same composition

For a singleton pixel profile, `pyrite run PROFILE --ephemeral --material MATERIAL`
resolves the physical beam, target, detector geometry, and ordered filters into
a physical `Scene`. The named detector's response, acquisition, and angular
sampling become an independent `Observation`; the profile's physics and
numerics are supplied separately to `simulate`. The profile's detector ID
is an observation label, not a source or observation identity field.

This normalization preserves the existing TOML schema, CLI JSON and artifact
contracts, and equivalent per-case numerical values. Batch CLI runs and the
scan app continue to use `build_configured_cases`, which resolves the
configured sweep through the same simulation-composition boundary while
retaining historical sweep ordering and checkpoint identities.

## Reuse bodies as targets and filters

`RectangularBody` defines reusable material and dimensions in **mm**;
`Target` and `Filter` give it different roles and geometry:

```python
from pyrite.detectors import PlanarPose
from pyrite.geometry import Filter, RectangularBody, SubstanceRef, Target

body = RectangularBody(
    substance=SubstanceRef(kind="crystal", key="silicon"),
    size_mm=(10.0, 10.0),
    thickness_mm=0.1,
)
source_target = Target(body=body, tilt_deg=30.0)
window = Filter(body=body, pose=PlanarPose((0.0, 0.0, 50.0)), name="window")
```

For existing flat slabs or stacks, use `Slab`/`Stack` directly. Filters are
primary-attenuation-only; the body does not itself determine the object's
role or placement.

`SubstanceRef(kind="crystal", key="silicon")` selects a catalog crystal;
`kind="medium"` selects a catalog medium. The body describes substance and
dimensions independently of placement. `Target` gives a crystalline body its
source-target orientation; `Filter` places a body downstream with a `PlanarPose`.
Reusing the body shares its material description without coupling these roles.

## Tilt the bunch front and crystal cut

`BunchFrontTilt` specifies a position–time correlation independently of the
beam's propagation direction. `Scene.phase_matched_bunch_front_tilt(...)`
returns a tilt that cancels the calculated transverse phase mismatch for a
particular reflection and observation direction; it does not modify the scene
or establish detector-integrated gain.

```python
from dataclasses import replace

from pyrite.beams import BunchFrontTilt

source = replace(
    beam,
    bunch_front_tilt=BunchFrontTilt(tilt_deg=45.0, azimuth_deg=0.0),
)
```

Supported analytic sources and approximations are described under
[coherent emission](../physics/radiation-physics/coherent-emission.md).

### Crystal cuts

Specify a flat crystal's lattice orientation relative to its surface with
`CrystalCut`, separately from the incident beam angle:

```python
from pyrite.geometry import CrystalCut, Footprint, Slab

cut = CrystalCut(surface_hkl=(1, 0, 1), roll_deg=17.0)
cut_target = Slab(
    "hopg",
    thickness_ang=500.0,
    tilt_deg=30.0,
    footprint=Footprint(5.0, 5.0),
    cut=cut,
)
```

`surface_hkl` (or, alternatively, `beam_uvw`) selects the lattice axis;
`roll_deg` and optional `miscut_deg`/`miscut_azim_deg` refine its orientation.
The cut moves with the slab under `tilt_deg` and `tilt_azim_deg`.
