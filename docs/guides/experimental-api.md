# Experimental Python API: lifecycle and admission

`experimental` is a **compatibility status**, not a scientific domain and not a
validation claim. The reviewed entry points are discoverable under
`pyrite.experimental.<domain>` and remain owned by their existing domains.
Existing `pyrite.radiation`, `pyrite.quantum`, and photon module imports continue
to refer to the **same objects**. Importing a facade is lightweight; resolving
an attribute may load the existing implementation and its dependencies.

```python
from pyrite.experimental.radiation import directional_sources
from pyrite.experimental.observations import load_photon_events
from pyrite.experimental.quantum import AxialDiagnosticConfig
```

Aliases do not change pickle module paths, dataclass identity, on-disk schemas,
numerical defaults, random streams, or simulation admission. They do not mean
all research diagnostics can be used in production `Case` instances.

## Compatibility and warnings

- Stable and advanced-stable APIs follow the supported compatibility and
  deprecation schedule. Experimental interfaces may evolve incompatibly in a
  minor release **after release-note notice and migration guidance**; they
  have no stable semver guarantee for argument or result shape.
- New experimental symbols need one owning domain, an intended stable import,
  explicit physical bounds, and entry-by-entry approval in `api-surface.toml`.
  Kernels, implicit wildcard exports, and private implementation details are
  not eligible merely because a research script calls them.
- Existing imports are not automatically deprecated or downgraded by an
  additional experimental alias. Do not warn on import or every simulation.
  Narrow runtime warnings require an actionable failure mode and review.
- API support is independent of *scientific* validation; neither an import
  alias nor an API promotion expands a model's physical validity.

## Initial triage (v0.21 research interfaces)

| Research interface | Domain owner / intended stable destination | Status, units and constraints | Facade in this slice |
| --- | --- | --- | --- |
| `directional_source(s)`, `DirectionalSpectrum` | `radiation` / `results` | Experimental detector-free signed angular-plane sampling. Escaped density **photons/(incident electron eV sr)**, not angular-bin counts; individual case/source constraints remain. | `experimental.radiation` |
| `simulate_photon_continuum`, `PhotonContinuumObservation`, continuum tally/save/load | `transport` / `observations` | Experimental weighted-photon continuum reference and physical-aperture observations. Reduced spectra are **photons/(incident electron eV)** integrated over aperture; saved tallies cannot restart tracks. | `experimental.observations` |
| `simulate_photon_events`, `PhotonEventObservation`, relaxation budget and event tally/save/load | `transport` / `observations` | Experimental physical-event reference with separately accounted energy budget. Event and continuum observations are not interchangeable; reduced artifacts are not checkpoints. | `experimental.observations` |
| `PhotonTransport` | `config` | Experimental configuration; accepted cases and feature combinations remain governed by lowering. | Not yet admitted |
| Scalar optical response from `materials.optical` | `materials` | Provisional optical/TR reference response: explicit source, interpolation intervals, isotropic scalar model, no material inference. | Not yet admitted |
| Density-matrix/MCWF solvers, axial diagnostics and planar source bridge | `channeling` (proposed) / `quantum` (current facade) | Experimental diagnostics. Only the existing planar density-matrix recipe is admitted to production; axial/coherent diagnostics are not. Runtime states are nonportable. | `experimental.quantum` pending channeling ownership review |
| `QuantumSetup`, `TransverseStateSpec`, runtime factories | `quantum` | **Advanced-stable** API contracts; scientific validation still separately bounded. | Intentionally excluded |

Physical parameter ranges, model assumptions, data restrictions, and backend
limitations remain documented in the existing [planar](experimental-planar-channeling.md),
[axial](experimental-axial-channeling.md), and [quantum](quantum-api.md)
guides. These aliases never extend those published restrictions.

## Admission and promotion checklist

1. **Classify:** identify one owning physical domain and future stable import.
   Record admitted energies, geometries, materials, backends, parameter ranges,
   units, output normalization, known failure modes and missing validation.
2. **Admit:** expose only approved identity aliases with typing declarations;
   record their exact underlying targets and `experimental` status in
   `api-surface.toml`. Test import-lightness, identity, and exact inventory.
3. **Validate science:** provide range-bounded physics tests, benchmarks,
   reference data and reproducibility requirements. API checks cannot replace
   this work.
4. **Review API:** explicitly sign off units, model/numerical boundaries,
   exceptions, identity/provenance/checkpoint effects, schema portability, and
   compatibility tests.
5. **Promote:** add the approved stable-domain import, update docs/inventory,
   release notes, and the existing compatibility/deprecation registry where
   appropriate. Do not silently remove existing imports or automatically
   promote from a successful research run. Give previously supported aliases
   a documented migration window.

Promotion is separate review work; import changes alone do not update science,
checkpoint compatibility, default numerics, or sample identity.
