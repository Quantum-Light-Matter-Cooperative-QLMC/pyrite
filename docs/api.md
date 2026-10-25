# Supported Python API

This page lists PyRITE's supported library objects. Their compatibility follows
semantic versioning: removal or an incompatible change requires a documented
deprecation or a major release. Unlisted public objects are provisional.
Underscore-prefixed names, CLI callbacks, app launchers, remote orchestration,
developer tools and compatibility-only re-exports are internal.

Start with the [Python API workflow](guides/python-api-workflow.md) for runnable
examples, units and result interpretation. The [CLI reference](repo-design/cli/cli-reference.md)
covers command-line contracts.

## Material authoring

Use `pyrite.materials.MaterialDefinition`, `build_medium`, `build_crystal` and
`save_material` to author materials. Builders validate without registration;
`save_material` persists and returns a fresh immutable catalog. Density is in
g/cm³, composition in atoms/Å³ and photon grids in eV. See
[Adding materials](guides/adding-materials.md) for inputs and storage rules.

## Photon-energy comparisons

{py:func}`pyrite.photon_energies` computes vacuum CXR line-centre energies
without transport. Beam and photon energies are in eV; angles are in degrees.
Broadcast mode pairs inputs; grid mode forms their Cartesian product. These
are kinematic energies, with no prediction of intensity or broadening. See
[comparison examples and reflection rules](guides/python-api-workflow.md#compare-photon-energies).

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.photon_energies
```

## Scene simulation

Build a beam, target and detector, then call {py:func}`pyrite.simulate` with
those three components, or pass a complete `Scene` as `simulate(scene)`.
Scene calls use its physics and numerics unless explicitly overridden with
`physics=` or `numerics=`; omit component arguments in that form. Target tilt
and azimuth retain full floating-point precision in scenes and profile sweeps.
`Physics` selects models and `Numerics` controls sampling and integration.
Beam transverse and
longitudinal descriptions and target footprints are explicit inputs.
{py:func}`pyrite.load_physics` and {py:func}`pyrite.load_numerics` return
named catalog presets (for example the bundled `electron-standard` and
`default`, or ones made with `pyrite physics|numerics preset create`) as
frozen `Physics` and `Numerics`; adjust them with `dataclasses.replace`
instead of restating every setting.

`simulate` returns an in-memory `Result` without reading or writing checkpoints.
Radiation families remain separate: `cxr.incoherent`, optional `cxr.coherent`,
`brem` and optional `characteristic`. Each spectrum carries its energy grid
and normalization. Scalar-detector densities are response-free photons per
incident electron per eV per sr; planar-detector densities are filtered,
solid-angle-weighted observation averages, still per sr. Use `spectral_flux()`
for current and acceptance scaling, or `acquire()` for registered pixel counts.

The workflow covers [results](guides/python-api-workflow.md#interpret-the-result),
[detector response](guides/python-api-workflow.md#apply-detector-response-explicitly),
[sweeps](guides/python-api-workflow.md#expand-a-path-addressed-sweep),
[adaptive sampling](guides/python-api-workflow.md#choose-adaptive-sampling) and
[shared target/filter bodies](guides/python-api-workflow.md#reuse-bodies-as-targets-and-filters).
For correlated imported beams, see [GDF beam import](guides/gpt-gdf-beams.md).

The provisional photon-continuum reference has a separate
[aperture tally artifact](research/physics/photon-scene-handoff.md#explicit-reduced-observation-persistence):
`PhotonContinuumObservation.save(path)` and
`pyrite.photon_observation.load_photon_continuum(path)` preserve reduced
aperture spectra and provenance. These experimental objects are outside the
supported API contract above. Their aperture-integrated spectra have units
photons/(incident electron eV) and must not be added to `Result.brem`.
The experimental low-level
[`score_photon_sources`](research/physics/photon-secondary-feedback.md#explicit-event-source-aperture-scoring)
boundary also scores explicit characteristic/annihilation source batches using
the original incident-electron denominator.
[`pyrite.photon_events.simulate_photon_events`](research/physics/photon-secondary-feedback.md#scene-physical-event-source-boundary)
selects one physical event component from a Scene with retained charged cascades.
Both remain outside the supported API contract. Physical event observations
have a separate [reduced event tally artifact](research/physics/photon-secondary-feedback.md#reduced-event-observation-persistence),
loaded with `pyrite.photon_event_observation.load_photon_events(path)`.
These analysis artifacts are not production checkpoints or replay inputs.

The experimental `Physics(photon_transport=PhotonTransport(...))` selector now
adds a separate `Result.aperture` observation through the ordinary scalar
simulation path, using one primary transport. `aperture_dataframe()` exposes
histogram densities and bin counts; `current_nA=` scales to photons/s without
reapplying acceptance. Enabled Case/checkpoint/CAS records preserve the aperture
and its original pins; full artifact replay is supported and bounded replay is
refused. See the [selection, units and persistence contract](repo-design/compute/photon-aperture.md).
`PhotonTransport` and the aperture APIs remain provisional; pixel response,
catalog/CLI selectors and full-track external validation remain open.

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
   pyrite.CrystalCut
   pyrite.PlanarPose
   pyrite.PixelGrid
   pyrite.FilterPlate
   pyrite.Acquisition
   pyrite.NativeAcquisition
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
   pyrite.load_physics
   pyrite.load_numerics
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

### Shared bodies and explicit roles

`RectangularBody` describes a solid in millimetres; `Target` and `Filter` assign
its scene role. See [body construction and limits](guides/python-api-workflow.md#reuse-bodies-as-targets-and-filters).

## Electron bunch-front tilt

`BunchFrontTilt` adds a position-dependent arrival time without rotating the
beam. A front parallel to the target entrance face need not phase-match
radiation in the observation direction. `Scene.phase_matched_bunch_front_tilt`
returns a front for one explicit reflection and direction on a flat single slab.
See [front timing, supported sources and examples](guides/python-api-workflow.md#tilt-the-bunch-front-and-crystal-cut).

### Crystal cuts

`CrystalCut` sets the lattice orientation relative to a flat slab's entrance
face. Target tilt moves face and lattice together. See the
[cut convention and limits](guides/python-api-workflow.md#crystal-cuts).

## Materials and crystallography

`pyrite.materials.CATALOG` is the immutable bundled `MaterialCatalog`.
`CRYSTALS`, `MATERIALS` and `MATERIAL_LABELS` are compatibility projections.
A material's display label derives from its `MaterialIdentity`; its cut names
the slab normal, independently of the reflections selected for scoring.

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

These lower-level entry points use the units in their docstrings. `Case` is
the frozen, validated input record; `to_dict()` provides its serialization
mapping. Prefer keyword arguments for optional model controls.

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
   pyrite.montecarlo.trajectory_summary.summarize_trajectories
   pyrite.montecarlo.trajectory_summary.iter_primary_summaries
   pyrite.montecarlo.trajectory_summary.depth_loss_profile
   pyrite.montecarlo.trajectory_balance.capture_secondary_energy_balance
   pyrite.campaign.beam_metrics.capture_beam_metrics
   pyrite.montecarlo.runner.run_case
   pyrite.montecarlo.runner.run_cases
   pyrite.montecarlo.channeling.spatial.spatial_density
   pyrite.montecarlo.channeling.axial_diagnostic.AxialDiagnosticConfig
   pyrite.montecarlo.channeling.axial_diagnostic.AxialDiagnosticResult
   pyrite.montecarlo.channeling.axial_diagnostic.axial_diagnostic
```

For planar channeling, `DensityMatrixSolver.propagate(spatial_x_ang=...)`
records density per incident electron per Å on the supplied grid. Its cell
integral equals the surviving solver trace. The recorded array has shape
`(len(z_ang), len(x_ang))`, includes entrance and exit, and folds density
into one period in the instantaneous basis after energy reprojection. Without
position sampling, both the grid and density are `None`. See the
[spatial-density derivation](validation/channeling/cr-planar-spatial-density-reconstruction.md)
and [channeling demo](guides/working-with-results.md#planar-channeling-demo).
The [experimental axial diagnostic](guides/axial-diagnostics.md) runs independently
of `simulate`.

## Detector response

Use these objects for detector efficiency, energy redistribution and grating
images. Physical pixel scoring and acquisition are illustrated in the
[workflow guide](guides/python-api-workflow.md#place-a-finite-filter-over-detector-pixels).

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

Single simulations return `Result`; resumable campaigns store result mappings
in [component checkpoints](repo-design/storage/dataset-identity-and-storage.md).
`Settings` is the compatibility surface for checkpoint analysis. `Analysis`
holds presentation controls and legacy conversions; `simulate` does not
consume it.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.results.Settings
   pyrite.results.records
   pyrite.results.filter_results
   pyrite.results.select_results
   pyrite.results.best_azimuth
   pyrite.results.cxr_metrics
   pyrite.results.radiation.radiation_band_metrics
   pyrite.results.summary_table
```

## Provisional directional source API

`pyrite.directional_source(beam, target, energy_eV=..., angle_deg=..., ...)`
returns `pyrite.DirectionalSpectrum` without a detector or checkpoint. The result
contains explicit coordinates, escaped incoherent PXR/CBS density per incident
electron/eV/sr, supported-direction mask and resolved provenance. `.rate()` uses
the beam mean current; `.rate(current_na)` rescales stored arrays to ph/s/eV/sr.
`pyrite.directional_sources(..., components=...)` returns one such result per
component (`"pxr-cbs"`, `"characteristic"`, `"bremsstrahlung"`) from one transport.
This new surface is provisional; see [directional heatmaps](guides/directional-source-heatmap.md)
for the bounded example, full-circle convention, supported geometry and limits.

## Provisional optical diagnostic response

`pyrite.materials.optical.OpticalResponse`, `gold_rakic_response`, and
`pyrite.energy_grid.component_support.{ComponentSupport,component_grid}` are
provisional source-only diagnostic APIs. They require explicit passive scalar
response and per-component supported bands; they do not lower the production
X-ray floor or supply optical detector/transport models. See
[explicit optical TR response](physics/radiation-physics/transition-radiation.md#explicit-optical-response-for-bounded-diagnostics)
for units, sample/provenance metadata, interpolation policy and an example.

`pyrite.detectors.optical_energy.CalibratedPulseEnergyDetector.predict_batch`
integrates collected photon densities shaped `(..., energy)` on a shared
nonuniform energy grid. Its `OpticalPulseBatchReading` carries per-pulse
joules, isolated-pulse volts, noise/saturation and optional average watts over
the batch axes. NumPy/CuPy inputs select the backend; `predict` retains the
single-spectrum interface returning host scalars. Calibration remains explicit
and bounded; these provisional operators supply no measured response by default.
`OpticalSpectralCoherency`, calibrated Jones transfers and
`polarized_interferogram` preserve that backend through the forward reductions.
See [device operators and transfer boundaries](physics/radiation-physics/optical-wavefront-and-detectors.md#device-resident-field-operators-and-cuda-parity).

Finite PEC source entry points in `pyrite.montecarlo.spectrum.optical_finite_screen`
and `optical_finite_bunch`, `optical_finite_collection.finite_screen_bunch_plane_band`,
and the three `runner.finite_screen_emission.case_finite_screen_*` adapters
accept an explicit `xp=np` (default) or active CuPy namespace. Meshes, Bessel
evaluation, complex fields and streamed self/pair moments stay on that backend;
screen objects, energy-node iteration, source-law metadata and formation guards
remain host control. `FiniteScreenBunchAngular.collect_cone(to_host=False)`
retains its scalar on device; the default returns a host float. Device band
arrays can feed `CalibratedPulseEnergyDetector.predict_batch` directly.
