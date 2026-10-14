# Supported Python API

The objects explicitly listed on this page are PyRITE's supported library surface. They follow semantic-versioning compatibility: removal or an incompatible signature/meaning change requires a documented deprecation or a major release. Public objects not listed here are usable but provisional.

Names beginning with `_`, CLI command callbacks, app launchers, remote-job orchestration, developer tools, and package-facade re-exports retained only for old imports are internal or compatibility surfaces. They may change without a library deprecation. The command-line contract is documented separately in the [CLI reference](repo-design/cli/cli-reference.md).

For a task-oriented walkthrough, including detector scoring and the boundary between in-memory results and checkpoint campaigns, see the [Python API workflow](guides/python-api-workflow.md).

For native GPT time-output and screen beams, see [GDF beam import](guides/gpt-gdf-beams.md).
Use `gdf_normalization="gdf_charge"` with the shared `rep_rate_hz` to derive
charge from selected records; this replaces configured `bunch_charge_pc`.
The default rate is 5000 Hz. Clear both analytic spot FWHMs for a GDF `Beam`.
Named catalog beams clear those inherited defaults during profile resolution.

## Material authoring

`pyrite.materials.MaterialDefinition`, `build_medium`, `build_crystal`, and
`save_material` are supported authoring APIs. Builders validate without
registration; save explicitly persists and returns a fresh immutable
`MaterialCatalog`. Density arguments are in g/cm³, composition values in
atoms/angstrom³, and crystal photon grids in eV. The returned definition has a
name-independent physical identity and an inline `MediumSpec` projection.
See [Adding materials](guides/adding-materials.md) for composition syntax,
CIF inputs, storage ownership, cache refresh and transport limits.

## Photon-energy comparisons

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

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.photon_energies
```

## Scene simulation

The root package exposes the preferred high-level API. A `Scene` contains one scalar `Beam`, target, and either the compatibility scalar `Detector` or a physical `PlanarDetector`; `Physics` contains emission and particle-interaction model choices. `Numerics` contains sampling, true-photon energy grids, integration, and execution controls. Both can be carried by `Scene` for sweeps. `simulate` lowers those objects to the established typed `Case` and calls the existing Monte Carlo runner directly. It neither reads nor writes a checkpoint.

Use `pr.Physics(brem_source=None)` to disable continuum scoring. The default is
`"mc"`; `"external"` selects an external continuum policy. `None` is normalized
to `"none"` internally, preserving case and checkpoint identity. The string
`"none"` remains accepted for compatibility.

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
    numerics=pr.Numerics(cxr_trials=450, brem_trials=100),
)
result.cxr.incoherent.density
result.provenance["identity_digest"]
```

`Result` groups radiation families: `result.cxr.incoherent`, optional
`result.cxr.coherent`, `result.brem`, and optional `result.characteristic`.
Each `SpectralDensity` pairs `photon_energy_eV` (eV) with `density` (photons per
incident electron per eV per sr) and its `normalization`. Brem may use a wider,
coarser grid than CXR. Characteristic emission uses the CXR grid but stays
separate from both CXR modes. Requesting coherent emission never changes the
meaning of `cxr.incoherent`; coherent and `both` requests expose the coherent
companion as well.

For a scalar `Detector`, normalization is `"source"`: response-free density
without acceptance, current, quantum efficiency, or measured-energy
redistribution. For a `PlanarDetector`, it is `"observation-average"`:
filter-attenuated, solid-angle-weighted density, still per sr and without
detector response. `result.combined_cxr(mode="incoherent")` returns a
`SpectralDensity` adding characteristic emission once; set
`include_characteristic=False` to return only the chosen CXR mode. It does not
add brem across different grids; `result.spectral_flux(units=...)` does, on the
union grid, in photons/(s eV) into the accepted solid angle or the other
`FLUX_UNITS`, with a `total` column. `result.to_dataframe()` returns the native
grids as a long polars frame. `Result.provenance` is a typed, read-only
`Provenance` mapping: inputs (`scene`, `numerics`), content identity, physics-model
markers, run diagnostics, and backend/device/version environment; see
[Interpret the result](guides/python-api-workflow.md#interpret-the-result).
`provenance.resolved_physics` records the resolved in-tree descriptor IDs,
model-generation markers, parameters, and per-case table digests. It is
descriptive metadata: adding a descriptor does not change source or observation
identity, and capability declarations do not certify a physical validation domain.
The high-level API
does not accept an external brem array: `none` and `external` return zeros on
the resolved continuum grid.

The flat Result attributes (`energy_eV`, `spectrum`, `background_energy_eV`,
`background`, `coherent_spectrum`, `characteristic_spectrum`), their positional
and flat-keyword constructor, and `line_total()` were removed after 0.11; use the
grouped families and `combined_cxr()`. `Result` takes keyword arguments only. `Numerics` uses
`cxr_trials`, `brem_trials`, and `cxr_chunk`; the old `n_electrons`,
`n_electrons_brem`, and `spec_chunk` keywords/attributes were removed in 0.13.
`EnergyBins.cxr`/`cxr_by_energy` and `Precision(observables=("cxr", "brem"))`
use the same vocabulary. Historical case/checkpoint/hash keys remain unchanged.

`simulate(..., physics=pr.Physics(emission=("incoherent", "coherent"), straggling=True))` separates
model choices from numerical accuracy. Sampled photon grids belong to
`pr.Numerics(energy_bins=pr.EnergyBins(cxr=..., brem=...))`; detector geometry
and read-time response do not own them. `Numerics.cxr_reflections` takes
`pr.CxrReflections(n_families=..., max_reflections=..., mosaic_nodes=..., mosaic_route=...)`.
Omitted grids still resolve from the catalog and built-in policy. Profile TOML
and CLI spellings are unchanged.

Removed in 0.14.0: `Convergence`, `Numerics(convergence=...)`,
model keywords on `Numerics`, `simulate(emission=..., brem_source=...)`, and
detector `energy_bins` keywords. Use `physics.<model>`, `numerics.energy_bins`,
and `numerics.cxr_reflections` for sweep paths; old ownership paths are rejected.
Beam transverse/longitudinal descriptions and target footprints are required.
Use `PointSpot`, `PlanarDetector.scorer`, and `Detector(acceptance=...)` in place
of the removed `Point`, loose `pixel_scorer`, and flat detector acceptance inputs.
Select `PYRITE_MC_BACKEND` before import rather than passing `Numerics.backend`.

For a physical pixel detector, `Acquisition` freezes exposure, explicit half-open measured-energy reporting edges, a post-response hit cut, and expected-versus-seeded-Poisson mode. `Result.acquire()` returns selected-pixel `AcquisitionBatch` records with registered reporting-bin counts and separate underflow, overflow, and below-cut accounting. `Result.acquisition_image()` contracts total or reporting-edge-aligned energy-window images in bounded pixel chunks. Expected counts use exposure, bunch charge, cadence, and the exact elementary charge; optional Poisson realizations are stable under selection and chunk-order changes. `ResolvedObservation` is the profile-lowered detector/scorer/filter/acquisition record. It deliberately projects back to the unchanged scalar `Detector` used by intrinsic transport.

`Sweep` expresses a Cartesian product as ordered paths into a scalar base scene. Paths are checked when the sweep is constructed, including indexed segments:

```python
sweep = pr.Sweep(
    base=pr.Scene(beam, target, detector),
    axes={
        "beam.energy_keV": [30.0, 45.0, 60.0],
        "target.tilt_deg": [15.0, 30.0, 45.0],
    },
)
cases = sweep.cases()  # uses base.numerics, then applies numerical axes
```

For a `Stack`, paths such as `target.layers[1].thickness_ang` address a particular layer. A misspelled field or out-of-range index raises at `Sweep` construction.

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

This mode supports incoherent scalar-detector scenes on the per-electron or CUDA core. It rejects coherent emission, physical detectors and observation directions, trajectory capture, GDF beams, grooves, and particle cascades. The two populations share one target and count in this implementation. The [statistical methods](computation/statistical-methods.md#adaptive-electron-counts) describe the observables, guards, and limits of the error estimate.

`Sweep.expand()` returns ordered `(label, Scene)` pairs; `Sweep.cases()` lowers them to typed cases but does not execute or persist them. `simulate` accepts the separate beam, target, and detector components, not a `Scene` argument. Helpers exported from `pyrite.api` but not listed on this page are provisional lowering or compatibility seams.

### Shared bodies and explicit roles

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
through a recipe to a different crystal is rejected. Crystal cuts remain
catalog-defined (`lattice_cut="catalog"`); arbitrary target poses, layers,
grooves and amorphous entrances are outside this adapter's supported set.
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

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.Beam
   pyrite.TransverseSpot
   pyrite.BunchFrontTilt
   pyrite.PointSpot
   pyrite.PointBunch
   pyrite.BunchLength
   pyrite.Offsets
   pyrite.GdfSource
   pyrite.Slab
   pyrite.Stack
   pyrite.SubstanceRef
   pyrite.RectangularBody
   pyrite.Target
   pyrite.Filter
   pyrite.Layer
   pyrite.Footprint
   pyrite.BlazedGrooves
   pyrite.PlanarPose
   pyrite.PixelGrid
   pyrite.FilterPlate
   pyrite.Acquisition
   pyrite.AcquisitionBatch
   pyrite.IdealPhotonCounter
   pyrite.PlanarDetector
   pyrite.PixelScorer
   pyrite.ResolvedObservation
   pyrite.PixelMetadata
   pyrite.PixelRayMap
   pyrite.SpectralFactors
   pyrite.SpatialResult
   pyrite.Scene
   pyrite.Sweep
   pyrite.Physics
   pyrite.CxrReflections
   pyrite.Numerics
   pyrite.Precision
   pyrite.Analysis
   pyrite.Result
   pyrite.Provenance
   pyrite.CxrSpectra
   pyrite.SpectralDensity
   pyrite.fetch_tables
   pyrite.simulate
```

## Electron bunch-front tilt

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
be multiplied unchanged. See the [independent derivation](validation/beam-transport/linear-bunch-front-tilt.md).
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
its mean catalog crystal orientation, rather than every mosaic orientation.
Target tilt and azimuth follow the existing nearest-half-degree case
resolution. The helper uses those resolved angles; bunch-front angles retain
full precision. Thus match the resolved entrance-face angles when setting a
parallel front manually.

Named or inline profile beam tables use the same descriptor:

```toml
[profiles.example.beam.bunch_front_tilt]
tilt_deg = 45.0
azimuth_deg = 0.0
```

## Materials and crystallography

`pyrite.materials.CATALOG` is the immutable bundled `MaterialCatalog`. `CRYSTALS`, `MATERIALS`, and `MATERIAL_LABELS` are compatibility projections, not independent registries and not part of the supported API.

Each `MaterialSpec` carries a `MaterialIdentity` — `formula`, `phase`, `full_name`, `cut`, `cut_frame` — and its `label` is derived from those fields rather than authored, so a display string cannot drift from the record it describes. The cut is the crystal's declared slab normal, reduced to its primitive representative; it is unrelated to the pinned `hkl_families` reflections.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.materials.MaterialCatalog
   pyrite.materials.MaterialConfigError
   pyrite.materials.MaterialIdentity
   pyrite.materials.CrystalInfo
   pyrite.materials.CrystalSpec
   pyrite.materials.MediumSpec
   pyrite.materials.MaterialSpec
   pyrite.materials.MaterialValidationSpec
   pyrite.materials.ScanSpec
   pyrite.materials.LayerSpec
   pyrite.materials.load_material_catalog
   pyrite.materials.atomic.cromer_mann_f0
   pyrite.materials.atomic.henke_dispersion
   pyrite.materials.atomic.atomic_form_factor
   pyrite.materials.crystal.reciprocal_g_vector
   pyrite.materials.crystal.structure_factor
   pyrite.materials.crystal.chi_g
   pyrite.materials.crystal.U_g
   pyrite.materials.crystal.absorption_length_ang
   pyrite.materials.crystal.dominant_reflections
   pyrite.materials.attenuation.linear_attenuation_inv_mm
```

## Simulation kernels

These are the supported low-level simulation entry points. Inputs and outputs use the units stated in their docstrings; prefer keyword arguments for optional model controls. `Case` is the frozen, validated input record; its `to_dict()` method preserves the legacy mapping representation for serialization.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.montecarlo.Case
   pyrite.montecarlo.transport.simulate_trajectories
   pyrite.montecarlo.spectrum.mc_cxr_spectrum
   pyrite.montecarlo.spectrum.mc_cxr_spectrum_solid_angle
   pyrite.montecarlo.spectrum.mc_brem_spectrum
   pyrite.detectors.response.detector_efficiency
   pyrite.detectors.response.convolve_detector
   pyrite.montecarlo.runner.run_case
   pyrite.montecarlo.runner.run_cases
```

## Detector response

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.detectors.Detector
   pyrite.AngularAcceptance
   pyrite.SensorAcceptance
   pyrite.detectors.EnergyBins
   pyrite.detectors.NativeSpectrum
   pyrite.detectors.Timepix3
   pyrite.detectors.EagleXO
   pyrite.detectors.LegacyEDS
   pyrite.detectors.eaglexo_response.EagleResponse
   pyrite.detectors.eaglexo_response.get_response
   pyrite.detectors.timepix_response.TimepixResponse
   pyrite.detectors.timepix_response.get_response
   pyrite.detectors.grating.Grating
   pyrite.detectors.grating.SimpleCCD
   pyrite.detectors.grating.disperse_spectrum
   pyrite.detectors.grating.detected_image
```

## Result analysis

Single-shot simulations return `pyrite.Result`. Resumable campaign checkpoints continue to use result dictionaries; their storage identity and lifecycle are documented in [dataset identity and storage](repo-design/storage/dataset-identity-and-storage.md). `Settings` is the D7 compatibility surface for those checkpoint-analysis functions. `Analysis` is the corresponding presentation-control value object and legacy-conversion target; it is not currently consumed by `simulate`.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.results.Settings
   pyrite.results.records
   pyrite.results.filter_results
   pyrite.results.select_results
   pyrite.results.best_azimuth
   pyrite.results.cxr_metrics
   pyrite.results.summary_table
```
