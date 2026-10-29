# Compatibility and canonical imports

The public API's stability classifications and canonical owner paths are
governed by `api-surface.toml`. Domain namespaces are preferred for specialized
classes; curated root imports remain supported as documented.

See the [experimental API lifecycle](../experimental-api.md) for shorter-lived contracts.

Aliases are object-identical to their canonical implementations; a new spelling
does not modify serialized values, numerical defaults, or physical validation.

Stable and advanced-stable compatibility follows
semantic versioning: removal or an incompatible change requires a documented
deprecation or a major release. Unlisted public objects are provisional.
Entries labeled experimental follow the shorter compatibility contract in the
[experimental API lifecycle](../experimental-api.md), which also records
domain owners, bounded feature contracts, and promotion criteria. Selected
experimental interfaces are discoverable under `pyrite.experimental.radiation`,
`pyrite.experimental.observations`, and `pyrite.experimental.quantum`.
Underscore-prefixed names, CLI callbacks, app launchers, remote orchestration,
developer tools and compatibility-only re-exports are internal.

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
normalizations (see the [spectral observables](../../physics/radiation-physics/spectral-observables.md)).

The [quantum API contracts](../quantum-api.md) documents portable specs, runtime
states, configuration, solver results, and experimental diagnostics through
`pyrite.quantum`. Existing quantum imports retain object identity, serialization,
and production admission.

Other namespaces retain their existing documented import paths. In particular,
`pyrite.config.KernelTuning` selects *local kernel controls* and is the **same
class** as `pyrite.config.ExecutionConfig` (retained for compatibility).
`Numerics(execution=KernelTuning(...))` and `with_kernel_tuning(...)` preserve
the historical lowering. This is not a scheduler profile
(`pyrite.execution.ExecutionProfile`). See the
[configuration ownership policy](../../repo-design/configuration-resolution.md#physics-numerics-and-kernel-tuning-702).
`pyrite.config.PhotonTransport` is experimental. See the
[API namespace policy](../../development/api-policy.md) for ownership and stability.

## Root import compatibility

The forward-looking root API is `Beam`, `Target`, `Detector`, `Scene`,
`Sweep`, `Physics`, `Numerics`, `simulate`, and `Result`, plus the approved
`photon_energies` convenience exception. Use domain imports for specialized types.

Only the names below are available through `dir(pyrite)`, static editor
declarations, and star imports. Specialized root aliases due in 0.22.0 have
been removed; use their documented domain imports. **Deferred** rows remain
supported at root without a warning or removal date until their ownership
decisions are resolved.
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
| `AcquisitionBatch` | Deferred | `pyrite.results.AcquisitionBatch` | Acquired pixel counts are result values; root migration awaits #707 facade approval. |
| `Analysis` | Deferred | `pyrite.campaign.model.Analysis` | Keep the existing import pending canonical ownership review in #707. |
| `Beam` | Retained | `pyrite.Beam` | Foundational simulation entry point. |
| `CxrSpectra` | Deferred | `pyrite.results.model.CxrSpectra` | Keep the existing import pending canonical ownership review in #707. |
| `Detector` | Retained | `pyrite.Detector` | Foundational simulation entry point. |
| `DirectionalSpectrum` | Deferred | `pyrite.results.directional.DirectionalSpectrum` | Keep the existing import pending canonical ownership review in #707. |
| `ExecutionConfig` | Deferred | `pyrite.config.ExecutionConfig` | Historical name for the same class. |
| `KernelTuning` | Deferred | `pyrite.config.KernelTuning` | Local kernel settings, not scheduling. |
| `Numerics` | Retained | `pyrite.Numerics` | Foundational simulation entry point. |
| `PhotonTransport` | Deferred | `pyrite.config.PhotonTransport` | Keep the experimental root path pending lifecycle review in #706. |
| `Physics` | Retained | `pyrite.Physics` | Foundational simulation entry point. |
| `PixelMetadata` | Deferred | `pyrite.results.model.PixelMetadata` | Keep the existing import pending canonical ownership review in #707. |
| `PixelRayMap` | Deferred | `pyrite.results.model.PixelRayMap` | Keep the existing import pending canonical ownership review in #707. |
| `Provenance` | Deferred | `pyrite.results.provenance.Provenance` | Keep the existing import pending canonical ownership review in #707. |
| `QuantumDynamics` | Deferred | `pyrite.quantum.QuantumDynamics` | Canonical quantum ownership is resolved in #704; retain the root alias without warnings pending a separate migration decision. |
| `QuantumNumerics` | Deferred | `pyrite.quantum.QuantumNumerics` | Canonical quantum ownership is resolved in #704; retain the root alias without warnings pending a separate migration decision. |
| `QuantumRegion` | Deferred | `pyrite.quantum.QuantumRegion` | Canonical quantum ownership is resolved in #704; retain the root alias without warnings pending a separate migration decision. |
| `QuantumSetup` | Deferred | `pyrite.quantum.QuantumSetup` | Canonical quantum ownership is resolved in #704; retain the root alias without warnings pending a separate migration decision. |
| `Result` | Retained | `pyrite.Result` | Foundational simulation entry point. |
| `Scene` | Retained | `pyrite.Scene` | Foundational simulation entry point. |
| `SpatialResult` | Deferred | `pyrite.results.model.SpatialResult` | Keep the existing import pending canonical ownership review in #707. |
| `SpectralDensity` | Deferred | `pyrite.results.model.SpectralDensity` | Keep the existing import pending canonical ownership review in #707. |
| `SpectralFactors` | Deferred | `pyrite.results.model.SpectralFactors` | Keep the existing import pending canonical ownership review in #707. |
| `Sweep` | Retained | `pyrite.Sweep` | Foundational simulation entry point. |
| `Target` | Retained | `pyrite.Target` | Foundational simulation entry point. |
| `TransverseStateSpec` | Deferred | `pyrite.quantum.TransverseStateSpec` | Canonical quantum ownership is resolved in #704; retain the root alias without warnings pending a separate migration decision. |
| `fetch_tables` | Deferred | `pyrite.xsgen.install.fetch_tables` | Keep the existing import pending canonical ownership review in #707. |
| `photon_energies` | Retained | `pyrite.photon_energies` | Approved deterministic line-energy convenience exception. |
| `simulate` | Retained | `pyrite.simulate` | Foundational simulation entry point. |

## Selected package contracts

The root inventory is supplemented by reviewed package and leaf-module names in
`api-surface.toml` under `package_exports`. In addition to material authoring,
the inventory retains the material, transport, detector-response, and analysis
contracts documented in these namespace pages as advanced-stable, with the axial diagnostic explicitly
experimental. It also includes `pyrite.observations.ObservationPlan`,
`pyrite.observations.PixelSampling`, `pyrite.results.Result`, and
`pyrite.results.Provenance`. A declaration in `__all__` alone is **not** an
API-support promise; consult the [API policy](../../development/api-policy.md)
for status and compatibility rules.
