# Supported Python API

This page lists PyRITE's supported library objects. Their compatibility follows
semantic versioning: removal or an incompatible change requires a documented
deprecation or a major release. Unlisted public objects are provisional.
Underscore-prefixed names, CLI callbacks, app launchers, remote orchestration,
developer tools and compatibility-only re-exports are internal.

Start with the [Python API workflow](guides/python-api-workflow.md) for runnable
examples, units and result interpretation. The [CLI reference](repo-design/cli/cli-reference.md)
covers command-line contracts.

## Canonical domain imports

New code can import the supported beam, geometry, and configuration objects
from their owning domains. The package-root aliases remain supported and are
**identical objects**; no physics defaults, serialization, or caching behavior
changes when choosing these imports.

```python
from pyrite.beams import Beam, TransverseSpot, BunchLength
from pyrite.geometry import Target, Slab, CrystalCut, RectangularBody, FilterPlate
from pyrite.config import Physics, Numerics, AngularSampling, load_physics
from pyrite.detectors import AngularDetector, PlanarDetector, PixelGrid, PlanarPose
from pyrite.detectors import Timepix3Response, IdealPhotonCounter
from pyrite.observations import Acquisition, Observation, PixelSampling, ObservationPlan
from pyrite.quantum import QuantumSetup, TransverseStateSpec
```

The **detectors** namespace owns virtual angular acceptance, physical pixel
geometry, and response models. `AngularDetector` is a semantic alias for the
legacy scalar `Detector`; `Timepix3Response` and `EagleXOResponse` are aliases
for existing response implementations, not new physical models. The
**observations** namespace owns immutable measurement requests (`Observation`,
`Acquisition`) and retains supported legacy plans and resolved types.
`pyrite.config.AngularSampling` is the immutable numerical scoring policy
formerly named `PixelScorer`; changing its `angular_shape` does *not* change
physical `PixelGrid.shape`. `pyrite.geometry.FilterPlate` owns the passive
filter plate role while the old detector import remains supported. Computed
`AcquisitionBatch` values canonically belong to `pyrite.results` (also
available as the descriptive `PixelCountResult` alias), with the older
observation/instrument imports preserved. These are object-identical
aliases: cache payloads, class pickle paths and numerical defaults do not change.

`Timepix3Response` supports the independent immutable
`pyrite.config.Timepix3ResponseNumerics` policy via its `numerics=` constructor
argument. Its four historical flat fields remain readable and constructible;
the underlying value and response identity are unchanged. Passing inconsistent
flat and grouped settings raises an error. `NativeBinnedResponse` is an optional
measured-energy-bin interface; a continuous `DetectorResponse.score()` alone
does not guarantee native bins. Event-level photon readout/digitization remains
a separate future capability, not implied by the factorized scorer.
A detector with a response and an `Acquisition` can request expected or
realized counts; intrinsic spectra remain photons/(incident electron eV sr),
whereas pixel-integrated and acquired results have different acceptances and
normalizations (see the [API workflow](guides/python-api-workflow.md)).

The [quantum API map](guides/quantum-api.md) documents portable specs, runtime
states, configuration, solver results, and experimental diagnostics through
`pyrite.quantum`. Existing quantum imports retain object identity, serialization,
and production admission.

Other namespaces retain their existing documented import paths. In particular,
`pyrite.config.KernelTuning` selects *local kernel controls* and is the **same
class** as `pyrite.config.ExecutionConfig` (retained for compatibility).
`Numerics(execution=KernelTuning(...))` and `with_kernel_tuning(...)` preserve
the historical lowering. This is not a scheduler profile
(`pyrite.execution.ExecutionProfile`). See the
[configuration ownership policy](repo-design/configuration-resolution.md#physics-numerics-and-kernel-tuning-702).
`pyrite.config.PhotonTransport` is experimental. See the
[API namespace policy](development/api-policy.md) for ownership and stability.

## Physical quantity inputs

Install the optional units extra to construct Pint quantities through PyRITE's
single shared registry. Existing numeric inputs keep the units encoded by their
field names. Supported configuration boundaries convert compatible quantities
to those same numeric values before case lowering or persistence.

```python
from pyrite.beams import Beam, PointBunch, Spot
from pyrite.observations import Acquisition
from pyrite.detectors import PixelGrid, PlanarPose
from pyrite.units import ureg

beam = Beam(
    energy_keV=0.06 * ureg.MeV,
    transverse=Spot(),
    longitudinal=PointBunch(),
)
pose = PlanarPose(center_mm=(0, 0, 10 * ureg.cm))
pixels = PixelGrid(shape=(256, 256), pitch_mm=(55 * ureg.um, 55 * ureg.um))
acquisition = Acquisition(
    exposure_s=500 * ureg.ms,
    measured_edges_eV=(0.1 * ureg.keV, 0.2 * ureg.keV, 0.3 * ureg.keV),
)
```

`from pyrite.units import ureg, Q_` requires `pip install 'pyrite-mc[units]'`.
The normal `import pyrite` path does not load Pint. Incompatible dimensions,
non-real magnitudes, and dimensionless values supplied to angular fields fail
at configuration construction. Stored values remain ordinary Python numbers
and NumPy arrays; quantities do not enter transport kernels or checkpoints.
See the [unit boundary and field inventory](repo-design/physical-units.md)
for supported fields, profile string syntax, serialization policy, and measured
overhead.

## Root import compatibility

The forward-looking root API is `Beam`, `Target`, `Detector`, `Scene`,
`Sweep`, `Physics`, `Numerics`, `simulate`, and `Result`, plus the approved
`photon_energies` convenience exception. Use domain imports for specialized types.

All existing names remain importable and available through `dir(pyrite)`, static
editor declarations, and legacy star imports. Deprecated root accesses emit
`DeprecationWarning` on each lookup (subject to Python warning filters); these
aliases are deliberately not cached. Canonical imports emit no root warning.
`__all__` preserves compatibility rather than defining the forward API.

**Deprecated** rows are scheduled for announcement in 0.20.5, with removal
targeted for 0.22.0. The release PR confirms the actual announcement version
and shifts the removal target if first shipped in a later minor.
Removal is separate release work. **Deferred** rows remain supported at root
without a warning or removal date until their ownership decisions are resolved.
`canonical` gives the recommended import today; deferred paths may be refined
by the linked follow-up. The governed classification is in `api-surface.toml`;
removal dates are owned by the existing deprecation registry.

`__version__` and `DATA_DIR` remain package metadata. The lazy `states` module
remains a supported compatibility exposure for the runtime factories now
available canonically from `pyrite.quantum`;
it is outside the domain-object allowlist. Logger and import plumbing are not
public API.

| Root name | Classification | Recommended import | Reason |
| --- | --- | --- | --- |
| `Acquisition` | Deprecated | `pyrite.observations.Acquisition` | Use the established canonical domain import. |
| `AcquisitionBatch` | Deferred | `pyrite.results.AcquisitionBatch` | Acquired pixel counts are result values; root migration awaits #707 facade approval. |
| `AdaptiveSampling` | Deprecated | `pyrite.config.AdaptiveSampling` | Use the established canonical domain import. |
| `Analysis` | Deferred | `pyrite.campaign.model.Analysis` | Keep the existing import pending canonical ownership review in #707. |
| `AngularAcceptance` | Deprecated | `pyrite.detectors.AngularAcceptance` | Use the established canonical domain import. |
| `Beam` | Retained | `pyrite.Beam` | Foundational simulation entry point. |
| `BlazedGrooves` | Deprecated | `pyrite.geometry.BlazedGrooves` | Use the established canonical domain import. |
| `BunchFrontTilt` | Deprecated | `pyrite.beams.BunchFrontTilt` | Use the established canonical domain import. |
| `BunchLength` | Deprecated | `pyrite.beams.BunchLength` | Use the established canonical domain import. |
| `CrystalCut` | Deprecated | `pyrite.geometry.CrystalCut` | Use the established canonical domain import. |
| `CxrReflections` | Deprecated | `pyrite.config.CxrReflections` | Use the established canonical domain import. |
| `CxrSpectra` | Deferred | `pyrite.results.model.CxrSpectra` | Keep the existing import pending canonical ownership review in #707. |
| `Detector` | Retained | `pyrite.Detector` | Foundational simulation entry point. |
| `DirectionalSpectrum` | Deferred | `pyrite.results.directional.DirectionalSpectrum` | Keep the existing import pending canonical ownership review in #707. |
| `EnergyBins` | Deprecated | `pyrite.detectors.EnergyBins` | Use the established canonical domain import. |
| `ExecutionConfig` | Deferred | `pyrite.config.ExecutionConfig` | Historical name for the same class. |
| `Filter` | Deprecated | `pyrite.geometry.Filter` | Use the established canonical domain import. |
| `FilterPlate` | Deprecated | `pyrite.geometry.FilterPlate` | Passive downstream geometry. |
| `FixedSampling` | Deprecated | `pyrite.config.FixedSampling` | Use the established canonical domain import. |
| `Footprint` | Deprecated | `pyrite.geometry.Footprint` | Use the established canonical domain import. |
| `GdfSource` | Deprecated | `pyrite.beams.GdfSource` | Use the established canonical domain import. |
| `IdealPhotonCounter` | Deprecated | `pyrite.detectors.IdealPhotonCounter` | Use the established canonical domain import. |
| `KernelTuning` | Deferred | `pyrite.config.KernelTuning` | Local kernel settings, not scheduling. |
| `Layer` | Deprecated | `pyrite.geometry.Layer` | Use the established canonical domain import. |
| `NativeAcquisition` | Deprecated | `pyrite.observations.NativeAcquisition` | Use the established canonical domain import. |
| `Numerics` | Retained | `pyrite.Numerics` | Foundational simulation entry point. |
| `Offsets` | Deprecated | `pyrite.beams.Offsets` | Use the established canonical domain import. |
| `PhotonTransport` | Deferred | `pyrite.config.PhotonTransport` | Keep the experimental root path pending lifecycle review in #706. |
| `Physics` | Retained | `pyrite.Physics` | Foundational simulation entry point. |
| `PixelGrid` | Deprecated | `pyrite.detectors.PixelGrid` | Use the established canonical domain import. |
| `PixelMetadata` | Deferred | `pyrite.results.model.PixelMetadata` | Keep the existing import pending canonical ownership review in #707. |
| `PixelRayMap` | Deferred | `pyrite.results.model.PixelRayMap` | Keep the existing import pending canonical ownership review in #707. |
| `PixelScorer` | Deprecated | `pyrite.config.PixelScorer` | Use `pyrite.config.AngularSampling` for new code; the legacy name remains the root replacement. |
| `PlanarDetector` | Deprecated | `pyrite.detectors.PlanarDetector` | Use the established canonical domain import. |
| `PlanarPose` | Deprecated | `pyrite.detectors.PlanarPose` | Use the established canonical domain import. |
| `PointBunch` | Deprecated | `pyrite.beams.PointBunch` | Use the established canonical domain import. |
| `PointSpot` | Deprecated | `pyrite.beams.PointSpot` | Use the established canonical domain import. |
| `Precision` | Deprecated | `pyrite.config.Precision` | Use the established canonical domain import. |
| `Provenance` | Deferred | `pyrite.results.provenance.Provenance` | Keep the existing import pending canonical ownership review in #707. |
| `QuantumDynamics` | Deferred | `pyrite.quantum.QuantumDynamics` | Canonical quantum ownership is resolved in #704; retain the root alias without warnings pending a separate migration decision. |
| `QuantumNumerics` | Deferred | `pyrite.quantum.QuantumNumerics` | Canonical quantum ownership is resolved in #704; retain the root alias without warnings pending a separate migration decision. |
| `QuantumRegion` | Deferred | `pyrite.quantum.QuantumRegion` | Canonical quantum ownership is resolved in #704; retain the root alias without warnings pending a separate migration decision. |
| `QuantumSetup` | Deferred | `pyrite.quantum.QuantumSetup` | Canonical quantum ownership is resolved in #704; retain the root alias without warnings pending a separate migration decision. |
| `RadiationNumerics` | Deprecated | `pyrite.config.RadiationNumerics` | Use the established canonical domain import. |
| `RectangularBody` | Deprecated | `pyrite.geometry.RectangularBody` | Use the established canonical domain import. |
| `ResolvedObservation` | Deprecated | `pyrite.observations.ResolvedObservation` | Use the established canonical domain import. |
| `Result` | Retained | `pyrite.Result` | Foundational simulation entry point. |
| `Scene` | Retained | `pyrite.Scene` | Foundational simulation entry point. |
| `SensorAcceptance` | Deprecated | `pyrite.detectors.SensorAcceptance` | Use the established canonical domain import. |
| `Slab` | Deprecated | `pyrite.geometry.Slab` | Use the established canonical domain import. |
| `SpatialResult` | Deferred | `pyrite.results.model.SpatialResult` | Keep the existing import pending canonical ownership review in #707. |
| `SpectralDensity` | Deferred | `pyrite.results.model.SpectralDensity` | Keep the existing import pending canonical ownership review in #707. |
| `SpectralFactors` | Deferred | `pyrite.results.model.SpectralFactors` | Keep the existing import pending canonical ownership review in #707. |
| `Spot` | Deprecated | `pyrite.beams.Spot` | Use the established canonical domain import. |
| `Stack` | Deprecated | `pyrite.geometry.Stack` | Use the established canonical domain import. |
| `SubstanceRef` | Deprecated | `pyrite.geometry.SubstanceRef` | Use the established canonical domain import. |
| `Sweep` | Retained | `pyrite.Sweep` | Foundational simulation entry point. |
| `Target` | Retained | `pyrite.Target` | Foundational simulation entry point. |
| `TransportNumerics` | Deprecated | `pyrite.config.TransportNumerics` | Use the established canonical domain import. |
| `TransverseSpot` | Deprecated | `pyrite.beams.TransverseSpot` | Use the established canonical domain import. |
| `TransverseStateSpec` | Deferred | `pyrite.quantum.TransverseStateSpec` | Canonical quantum ownership is resolved in #704; retain the root alias without warnings pending a separate migration decision. |
| `directional_source` | Deprecated | `pyrite.radiation.directional_source` | Canonical radiation import; remains experimental. |
| `directional_sources` | Deprecated | `pyrite.radiation.directional_sources` | Canonical radiation import; remains experimental. |
| `fetch_tables` | Deferred | `pyrite.xsgen.install.fetch_tables` | Keep the existing import pending canonical ownership review in #707. |
| `load_numerics` | Deprecated | `pyrite.config.load_numerics` | Use the established canonical domain import. |
| `load_physics` | Deprecated | `pyrite.config.load_physics` | Use the established canonical domain import. |
| `photon_energies` | Retained | `pyrite.photon_energies` | Approved deterministic line-energy convenience exception. |
| `simulate` | Retained | `pyrite.simulate` | Foundational simulation entry point. |


## Experimental optimization

The provisional `pyrite.optimization` API operates above immutable Scenes and
existing executors. It is outside the supported compatibility contract below.
See [optimization studies](guides/optimization-studies.md) for persistence,
constraints, normalization and the fixed-window PXR/CBS example.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.optimization.FixedWindowYield
   pyrite.optimization.PyriteEvaluator
   pyrite.optimization.OptimizationStudy
```

## Material authoring

Use `pyrite.materials.MaterialDefinition`, `build_medium`, `build_crystal` and
`save_material` to author materials. Builders validate without registration;
`save_material` persists and returns a fresh immutable catalog. Density is in
g/cm³, composition in atoms/Å³ and photon grids in eV. See
[Adding materials](guides/adding-materials.md) for inputs and storage rules.

The provisional `pyrite.materials.optical.OpticalResponse.to_record()` and
`OpticalResponse.from_record(record)` preserve explicit optical response
arrays, disjoint supported bands and provenance through JSON round trips.
They validate the existing content digest without reloading source data.
These response records remain outside the supported API contract and do not
enable optical production transport; see the
[proposed Cherenkov integration boundary](research/physics/production-cherenkov.md#identity-and-portable-response-records).

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
The canonical `Scene` dataclass fields are the physical `beam`, `target`,
`detector` geometry and ordered `filters`. Legacy constructor inputs for
`physics`, `numerics`, `quantum` and `acquisition` remain accepted through a
compatibility adapter, but are excluded from the physical dataclass value and
its `asdict()` representation. Prefer passing `physics=` and `numerics=` to
`simulate()` and acquisition through `Observation`. Pass an independent
{py:class}`pyrite.quantum.QuantumSetup` with `quantum=`; omission inherits a
legacy Scene setup and explicit `quantum=None` disables it.
`pyrite.api.build_case` uses the same precedence and validation rules, and
unsupported or contradictory quantum settings fail before case preparation or
transport. Effective quantum solver controls are recorded in result numerics.
Target tilt
and azimuth retain full floating-point precision in scenes and profile sweeps.
`Physics` selects models and `Numerics` controls sampling and integration.
Beam transverse and
longitudinal descriptions and target footprints are explicit inputs.
{py:func}`pyrite.config.load_physics` and {py:func}`pyrite.config.load_numerics` return
named catalog presets (for example the bundled `electron-standard` and
`default`, or ones made with `pyrite physics|numerics preset create`) as
frozen `Physics` and `Numerics`; adjust them with `dataclasses.replace`
instead of restating every setting.

Pass {py:class}`pyrite.observations.Observation` with `observation=` to select
response, acquisition and pixel sampling independently of the scene. Its
`detector="detector"` reference selects `Scene.detector`; a supplied planar
value must match that physical geometry. A scalar `Detector` value requests a
virtual angular observation, including on a detector-free source scene.
`sampling=` currently uses the supported `pyrite.observations.PixelScorer`.
Omitted observation fields inherit legacy embedded settings. Differing explicit
settings raise; start from a detector without embedded response/scorer and a
scene without acquisition to reuse one physical definition across observations.
Names are labels and do not change scientific identity.
`result.provenance["input_scene"]` retains the original scene for both legacy
and independently composed calls. The `"physics"`, `"quantum"` and
`"acquisition"` provenance keys expose the effective independent configuration;
`"numerics"` retains the effective numerical controls. These values do not
change case or observation identity. Meanwhile,
`result.scene` remains the resolved compatibility view used by read-time methods.
The [observation example](guides/python-api-workflow.md#compose-independent-observations)
shows how to reuse the same geometry.

Use `simulate(scene, executor=executor)` to select an existing
`pyrite.execution.Executor` independently of the physical scene. For example,
`LocalExecutor()` uses the established batch runner for the single case.
It can start CPU worker processes, so scripts should invoke it inside an
`if __name__ == "__main__":` guard, as with other batch-runner calls.
Omitting `executor` retains the original in-process simulation path. Scalar,
planar and counting observations use the same result assembly and scientific
identity in both forms. The executor must return one completed result;
interrupted or missing directional results fail explicitly.

This opt-in route currently requires `Numerics.transport_core="auto"`, since
batch executors do not accept a per-call transport-core override. Kernel
chunk settings remain on `Numerics`; scheduler settings remain on the executor.
Dask and SLURM executors keep their existing caller-configured artifact stores
and table-aware case keys. Their returned artifacts are checked against the
resolved source key and requested directions before use. Such executors can
write worker artifacts; `simulate` does not write simulation checkpoints.
`result.provenance["execution"]` records the executor class and its available
typed resource, scheduler and retry settings separately from scientific
identity. These describe requests, not measured worker hardware. A shared scientific identity does not promise bitwise
agreement across backends or scheduling strategies.

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
Both photon Scene entry points accept independent `physics=`, `numerics=`,
`quantum=` and `observation=` through the shared simulation resolver. Omitted
quantum inherits legacy Scene settings; `quantum=None` disables that setup.
Their ideal, unpixelated aperture restrictions still apply to observations.
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
   pyrite.beams.TransverseSpot
   pyrite.beams.BunchFrontTilt
   pyrite.beams.PointSpot
   pyrite.beams.PointBunch
   pyrite.beams.BunchLength
   pyrite.beams.Offsets
   pyrite.beams.GdfSource
   pyrite.geometry.Slab
   pyrite.geometry.Stack
   pyrite.geometry.SubstanceRef
   pyrite.geometry.RectangularBody
   pyrite.Target
   pyrite.geometry.Filter
   pyrite.geometry.Layer
   pyrite.geometry.Footprint
   pyrite.geometry.BlazedGrooves
   pyrite.geometry.CrystalCut
   pyrite.detectors.PlanarPose
   pyrite.detectors.PixelGrid
   pyrite.detectors.FilterPlate
   pyrite.observations.Acquisition
   pyrite.observations.Observation
   pyrite.observations.NativeAcquisition
   pyrite.observations.AcquisitionBatch
   pyrite.detectors.IdealPhotonCounter
   pyrite.detectors.PlanarDetector
   pyrite.observations.PixelScorer
   pyrite.observations.ResolvedObservation
   pyrite.PixelMetadata
   pyrite.PixelRayMap
   pyrite.SpectralFactors
   pyrite.SpatialResult
   pyrite.Scene
   pyrite.Sweep
   pyrite.Physics
   pyrite.config.load_physics
   pyrite.config.load_numerics
   pyrite.config.CxrReflections
   pyrite.Numerics
   pyrite.config.Precision
   pyrite.quantum.TransverseStateSpec
   pyrite.quantum.QuantumSetup
   pyrite.quantum.QuantumRegion
   pyrite.quantum.QuantumDynamics
   pyrite.quantum.QuantumNumerics
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

(simulation-kernels)=
## Advanced transport and radiation

Use `pyrite.transport` for trajectory simulation and saved-capture reductions;
use `pyrite.radiation` for source-level spectra. These lazy facades resolve to
exactly the existing implementation objects: signatures, errors, numerical
outputs, RNG streams, provenance, serialization and scientific identities are
unchanged. Importing either namespace loads no engines or material datasets;
resolving a function loads its implementation, and calling it may read tables,
select an accelerator or perform the documented artifact I/O.

```python
from pyrite.transport import Case, simulate_trajectories, summarize_trajectories
from pyrite.radiation import photon_energies, mc_cxr_spectrum, mc_brem_spectrum

# Provisional directional API; no stability promotion:
from pyrite.radiation import directional_source, directional_sources
from pyrite.results.directional import DirectionalSpectrum
```

`pyrite.simulate(scene)` remains the supported high-level one-scene operation.
`Case` is the frozen, validated lowered record with `to_dict()` serialization;
it remains advanced-stable for existing advanced callers. Prefer `Scene` for
new scene construction. No engine package becomes stable as a whole.

### Scientific contracts and migration

All transport facade entries below are **advanced-stable**. Units in keyword
and table-column names are literal: angstroms, keV and radians. Optional model
controls should be passed by keyword. The linked function docstrings define
full defaults, supported combinations and exceptions.

| Canonical import (`pyrite.transport.`) | Previous import | Input/output and effects |
| --- | --- | --- |
| `Case` | `pyrite.montecarlo.Case` | Validated lowered simulation/configuration record; field-specific units, frozen mapping serialization. Invalid fields raise validation errors; no transport on construction. |
| `simulate_trajectories` | `pyrite.montecarlo.transport.simulate_trajectories` | Incident energy/cutoff in keV, thickness/segment positions and lengths in angstroms; returns segment arrays and incident `Ne`/fate metadata. Samples transport with the supplied seed; may resolve elastic/stopping tables and backend. Unsupported model combinations/table coverage raise errors; incomplete step-limited histories raise. No detector response or photon tally. |
| `summarize_trajectories` | `pyrite.montecarlo.trajectory_summary.summarize_trajectories` | Reads a saved capture and explicit depth/energy/angle edges; returns histogram DataFrames. Fate fractions divide by stored incident `Ne`, including missed entries. Depth loss is keV/bin without primary, mass or volume normalization. Invalid edges/missing row fields raise. |
| `iter_primary_summaries` | `pyrite.montecarlo.trajectory_summary.iter_primary_summaries` | Streams capture DataFrames: primary fate, depth/range (angstrom), terminal pre-event energy (keV), angle (rad). Descendants do not determine primary fate; missed entries have no material rows. Artifact validation and file errors propagate. |
| `depth_loss_profile` | `pyrite.montecarlo.trajectory_summary.depth_loss_profile` | Stored segment mapping plus depth edges (angstrom) to continuous-loss keV/bin by layer. Excludes cutoff residuals/hard transfers; no primary normalization or file I/O. Invalid edges raise. |
| `capture_secondary_energy_balance` | `pyrite.montecarlo.trajectory_balance.capture_secondary_energy_balance` | Reads secondary-transport capture, reuses its canonical energy ledger; returns `term, energy_keV`. Requires secondary metadata and positive block limit; propagates artifact/file errors. No resimulation. |

Radiation energy grids are in **eV**; low-level segment electron energies are
in **keV**. Directions use the frame and angle unit specified by each function.

| Canonical import (`pyrite.radiation.`) | Status | Previous import | Input/output and effects |
| --- | --- | --- | --- |
| `photon_energies` | Advanced-stable | `pyrite.photon_energies` (retained convenience), `pyrite.materials.emission.photon_energies` | Catalog materials, electron energy (eV), geometry angles (degrees); returns vacuum line centres (eV), with broadcast or Cartesian axes. Reads catalog when called; no transport/yield/provenance result. Invalid values, unresolved cuts or catalog keys raise. |
| `mc_cxr_spectrum` | Advanced-stable | `pyrite.montecarlo.spectrum.mc_cxr_spectrum` | Segments, crystal/reflections, photon grid (eV), direction/angles (rad); escaped CXR density in ph/incident-electron/eV/sr, optionally separate PXR/CBS components. Uses stored incident population or explicit electron limit, with existing coherent population controls; reads material response and may use accelerator arrays. Unsupported coherence/geometry/quadrature combinations raise. |
| `mc_cxr_spectrum_solid_angle` | Advanced-stable | `pyrite.montecarlo.spectrum.mc_cxr_spectrum_solid_angle` | Same source inputs plus directions and weights (sr); returns ph/incident-electron/eV, already integrated over solid angle. Shape mismatch or tiled grooves raise. Never multiply by solid angle again. |
| `mc_brem_spectrum` | Advanced-stable | `pyrite.montecarlo.spectrum.mc_brem_spectrum` | Segments, photon grid (eV), composition (atoms/angstrom³), direction/angles (rad); escaped incoherent density in ph/incident-electron/eV/sr. Existing cutoff/population and cross-section selection remain in force; reads cross-section/material tables as needed; invalid models/inputs or missing required coverage raise. |
| `directional_source`, `directional_sources` | Experimental | `pyrite.directional.directional_source`, `pyrite.directional.directional_sources`; deprecated root aliases | Public scalar beam and flat target, explicit grid (eV), periodic angle points (degrees), fixed sampling and seed. One shared transport returns one `DirectionalSpectrum` per component with ph/incident-electron/eV/sr, case, provenance and support mask; unsupported infinite-slab grazing points are masked NaN. Invalid grids/angles, adaptive precision and grooves raise. |

The low-level spectrum arrays carry no new provenance wrapper: retain their
input segments, table/model choices and normalization metadata. Directional
results retain their existing case/provenance and current-scaled `rate()`;
`DirectionalSpectrum` remains an experimental **results** type at
`pyrite.results.directional`. No additional result class is introduced here.
These escaped densities include source self-absorption; they are distinct from
transported photon events, aperture tallies and acquisition counts. Neither
facade applies detector efficiency, electronic response or exposure.

Previous implementation imports remain supported compatibility paths. Only the
two root directional aliases enter the existing versioned deprecation registry;
use their canonical radiation imports. Future kernel exports require explicit
review and manifest entries; `montecarlo.__all__` is frozen by its export guard.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.transport.Case
   pyrite.transport.simulate_trajectories
   pyrite.radiation.mc_cxr_spectrum
   pyrite.radiation.mc_cxr_spectrum_solid_angle
   pyrite.radiation.mc_brem_spectrum
   pyrite.detectors.response.detector_efficiency
   pyrite.detectors.response.convolve_detector
   pyrite.transport.summarize_trajectories
   pyrite.transport.iter_primary_summaries
   pyrite.transport.depth_loss_profile
   pyrite.transport.capture_secondary_energy_balance
```

### Other documented engine operations

These remain at their current owners; listing them does not promote other
engine internals. All retain their existing support level.

| Entry | Owner / status | Scientific contract and effects |
| --- | --- | --- |
| `detector_efficiency`, `convolve_detector` | Detectors, advanced-stable | Photon grid (eV), window thicknesses (nm) to dimensionless efficiency; Gaussian FWHM (eV) convolution preserves spectral-density units. Optional outside mass has density × eV units. Reads attenuation tables when efficiency is evaluated; invalid grid/array inputs propagate numerical errors. Neither function applies an exposure. |
| `capture_beam_metrics` | Beam diagnostics, advanced-stable | Reads a bounded deterministic prefix of stored initial particles (including missed entries); returns beam metrics plus sampled/available counts. Metric field units follow `BeamMetrics`; invalid prefix limits or missing artifact data raise. No new transport. |
| `run_case`, `run_cases` | Runner orchestration, advanced-stable compatibility paths | Lowered `Case`/case mappings, field-specific units; execute transport and scoring, resolving tables/backend and retaining existing result/provenance contracts. Batch scheduling/collection and existing errors remain runner concerns; use `simulate` for public scene orchestration. No new transport facade aliases. |
| `spatial_density` | Quantum/channeling, advanced-stable | Density matrix, basis eigenfunctions and spatial grid (angstrom); Born-rule probability density (1/angstrom), including retained survival weight, without conditioning or renormalization. Array compatibility errors propagate; no transport or artifact I/O. |
| `AxialDiagnosticConfig`, `AxialDiagnosticResult`, `axial_diagnostic` | Quantum/channeling, experimental | Bounded solver configuration (field-specific units), populations and incident-normalized spatial density on recorded depths; retained solver/configuration diagnostics. Explicit solver/allocation limits and invalid configurations raise. No standard axial transport integration. Ownership consolidation belongs to #704. |

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

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
The [experimental axial diagnostic](guides/experimental-axial-channeling.md) runs independently
of `simulate`.

## Detector response

Use these objects for detector efficiency, energy redistribution and grating
images. Physical pixel scoring and acquisition are illustrated in the
[workflow guide](guides/python-api-workflow.md#place-a-finite-filter-over-detector-pixels).

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.detectors.Detector
   pyrite.detectors.AngularAcceptance
   pyrite.detectors.SensorAcceptance
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

`pyrite.radiation.directional_source(beam, target, energy_eV=..., angle_deg=..., ...)`
returns `pyrite.results.directional.DirectionalSpectrum` without a detector or checkpoint. The result
contains explicit coordinates, escaped incoherent PXR/CBS density per incident
electron/eV/sr, supported-direction mask and resolved provenance. `.rate()` uses
the beam mean current; `.rate(current_na)` rescales stored arrays to ph/s/eV/sr.
`pyrite.radiation.directional_sources(..., components=...)` returns one such result per
component (`"pxr-cbs"`, `"characteristic"`, `"bremsstrahlung"`, `"transition"`)
from one transport. The opt-in `"transition"` component uses the production
history-coherent planar-crossing model; it requires `footprint=None` and the
supported X-ray dielectric band. Cherenkov remains an internal bulk diagnostic
in the app, outside this escaped-source API.
This new surface is provisional; see [directional heatmaps](guides/directional-source-heatmap.md)
for the bounded example, full-circle convention, supported geometry and limits.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.radiation.directional_source
   pyrite.radiation.directional_sources
   pyrite.results.directional.DirectionalSpectrum
```

## Provisional optical diagnostic response

`pyrite.materials.optical.OpticalResponse`, `gold_rakic_response`, and
`pyrite.energy_grid.component_support.{ComponentSupport,component_grid}` are
provisional source-only diagnostic APIs. They require explicit passive scalar
response and per-component supported bands; they do not lower the production
X-ray floor or supply optical detector/transport models. See
[explicit optical TR response](physics/radiation-physics/transition-radiation.md#explicit-optical-response-for-bounded-diagnostics)
for units, sample/provenance metadata, interpolation policy and an example.

`pyrite.materials.optical.fused_silica_franta_response()` provides pinned
fitted fused-silica complex-index data with absorption retained.
`OpticalResponse.require_band(lower_eV, upper_eV)` requires whole-interval
coverage within one segment, preventing integration across unsupported gaps.
These provisional inputs do not establish production Cherenkov validity or
detector calibration; see the [reference-input contract](research/physics/production-cherenkov.md#fused-silica-reference-input).

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

## Selected package contracts

The root inventory is supplemented by reviewed package and leaf-module names in
`api-surface.toml` under `package_exports`. In addition to material authoring,
the inventory retains the material, transport, detector-response, and analysis
contracts listed above as advanced-stable, with the axial diagnostic explicitly
experimental. It also includes `pyrite.observations.ObservationPlan`,
`pyrite.observations.PixelSampling`, `pyrite.results.Result`, and
`pyrite.results.Provenance`. A declaration in `__all__` alone is **not** an
API-support promise; consult the policy for status and compatibility rules.
