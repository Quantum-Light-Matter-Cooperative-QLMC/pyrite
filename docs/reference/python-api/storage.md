# Persistence and storage

`pyrite.storage` owns supported checkpoint, observation and worker-artifact
persistence operations. Scientific outputs belong to
[`pyrite.results`](results.md), while measurement requests belong to
[`pyrite.observations`](observations.md).

Consult [result/storage contracts](../../repo-design/result-storage-contracts.md)
for artifact identities and format guarantees, and
[trajectory workflows](../trajectory-workflows.md) for persisted captures.

[Distributed worker artifacts](../distributed-artifacts.md) specify publication,
integrity checks and coordinator-owned checkpoint integration.

## Component checkpoints

Resolve checkpoint locations and inspect or load persisted components.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.storage.checkpoint_manifest
   pyrite.storage.checkpoint_path_for
   pyrite.storage.default_checkpoint_dir
   pyrite.storage.load_checkpoint
```

## Stored observations

Persist measurement observations separately from simulation results.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.storage.ObservationStore
   pyrite.storage.ObservationStoreError
   pyrite.storage.StoredObservation
   pyrite.storage.default_observation_root
   pyrite.storage.observation_from_result
```

## Worker artifacts

Publish and verify immutable worker artifacts and load directional reductions.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.storage.ArtifactIntegrityError
   pyrite.storage.CaseArtifactRef
   pyrite.storage.CaseArtifactStore
   pyrite.storage.DirectionalArtifactRef
   pyrite.storage.directional_content_key
   pyrite.storage.load_directional_result
```
