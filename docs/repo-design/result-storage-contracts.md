# Result values and storage contracts (#707)

PyRITE separates **scientific values** from **artifact persistence**. Imports
under `pyrite.results` give access to computed spectra, provenance, metrics and
reductions. `pyrite.storage` is the canonical entry point for supported
persistence operations; the existing implementations continue to own the
on-disk formats and their identities. No conversion or migration is needed.

## Ownership and identity

| Concept | Canonical owner | Representation, normalization and identity |
| --- | --- | --- |
| Intrinsic scalar result | `pyrite.results.Result` | `CxrSpectra`, `SpectralDensity`, continuum and optional radiation components; source density in photons / (incident electron eV sr); `Provenance.identity_digest` names the source calculation. |
| Directional spectrum | `pyrite.results.directional` | Direction-dependent **source** spectra, not a detector acceptance integral. Direction samples contribute to their own content identity when persisted. |
| Aperture tally | `Result.aperture` | Hit-integrated photons / (incident electron eV), with a separate histogram grid. **Do not multiply by a solid angle again.** |
| Physical pixel factors | `pyrite.results.SpatialResult`, `SpectralFactors`, `PixelRayMap`, `PixelMetadata` | Pixel mapping, solid angle, filter paths and factorized intrinsic spectra; no materialized pixel-by-energy cube. `true_spatial_digest` includes source identity and spatial scoring geometry. |
| Pixel observation | `pyrite.storage.StoredObservation` / `ObservationStore` | Separately identified true-spatial, detector-response and acquisition layers. `observation_identity_digest` names the complete observation; source and response digests alone do not. |
| Acquired pixel counts | `pyrite.results.AcquisitionBatch` (`PixelCountResult`) | Post-response/exposure expected or realized counts, **not** source flux density. Existing class/pickle identity remains `pyrite.instrument.acquisition.AcquisitionBatch`. |
| Campaign checkpoint | `pyrite.storage.load_checkpoint`, `checkpoint_manifest` | Per-material `{name: {E0_keV: record}}` store. Campaign orchestration owns writes, shard consolidation, resumability and manifest publication. |
| Distributed worker artifact | `pyrite.storage.CaseArtifactStore`, `CaseArtifactRef` | Immutable pre-`store_result` worker output, with separately validated **physics content key** and **byte digest**. Does not replace an authoritative campaign checkpoint. |
| Legacy result dictionary and analysis | `pyrite.results.store`, `pyrite.results.selection`, `pyrite.results.tables` | `store_result` creates an **in-memory record**, not a storage write. `Settings` and `Analysis` are historical display/reporting choices; neither is part of `simulate()`'s physics request. |

`Provenance` belongs to `pyrite.results` and is carried into relevant scientific
objects. It records, without rehashing, the source and observation identities,
model names, software versions, backend and device where available.

There are **three different kinds of identity** in the persistent pipeline:
(1) the transport/source content key, (2) the layered observation digests,
and (3) the worker artifact's byte-level SHA-256 receipt. Do not treat one as
another, nor merge their namespaces. Exposure, response, and beam normalization
may change a stored observation's identity without changing source spectra.

## Canonical imports

```python
from pyrite.results import Result, Provenance, AcquisitionBatch
from pyrite.storage import (
    CaseArtifactStore,
    ObservationStore,
    checkpoint_manifest,
    load_checkpoint,
    observation_from_result,
)

# Stored campaign results remain conventional dictionaries.
records = load_checkpoint("hopg", checkpoint_dir="./pyrite-output/checkpoints")
manifest = checkpoint_manifest("hopg", checkpoint_dir="./pyrite-output/checkpoints")

# Write and reopen a factorized observation from an acquisition-enabled Result.
# This is independent of the campaign checkpoint lifecycle.
def save_observation(result: Result, root: str) -> str:
    store = ObservationStore("hopg", root=root)
    return store.put(observation_from_result(result))

# Distributed executors publish independent raw case outputs, not checkpoints.
artifacts = CaseArtifactStore("./pyrite-output/checkpoints")
```

The storage facade is intentionally **not** a generic `save_result()` entry
point. A one-shot `Result` does not automatically contain a persistable pixel
observation, and publishing a campaign checkpoint requires coordinator-owned
completion and resume semantics. Use `ObservationStore.put()` for an
acquisition-enabled result, `CaseArtifactStore.publish()` for an immutable
worker output, or the normal sweep runner for campaign checkpoints.

The existing `pyrite.checkpoints.persistence`, `pyrite.observations.store`, and
`pyrite.execution.artifacts` imports continue to resolve to exactly the same
objects. Their serialized class module names, artifact schemas, checkpoint
manifests, cache keys and normalization conventions do not change.

## Future separation

This first slice establishes the public storage boundary. Follow-up work can
review the legacy `results.__all__` star-import footprint and migrate the
historical `Analysis` display controls to explicit transforms. Such changes
must preserve the published deprecation schedule; they must not silently
alter scientific factors, normalization, checkpoint data or cache identities.
