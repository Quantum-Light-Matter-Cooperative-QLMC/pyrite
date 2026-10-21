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
those three components. `Scene` collects them for sweeps; `Physics` selects
models and `Numerics` controls sampling and integration. Beam transverse and
longitudinal descriptions and target footprints are explicit inputs.

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
   pyrite.results.summary_table
```

## Provisional directional source API

`pyrite.directional_source(beam, target, energy_eV=..., angle_deg=..., ...)`
returns `pyrite.DirectionalSpectrum` without a detector or checkpoint. The result
contains explicit coordinates, escaped incoherent PXR/CBS density per incident
electron/eV/sr, supported-direction mask and resolved provenance. `.rate()` uses
the beam mean current; `.rate(current_na)` rescales stored arrays to ph/s/eV/sr.
This new surface is provisional; see [directional heatmaps](guides/directional-source-heatmap.md)
for the bounded example, full-circle convention, supported geometry and limits.
