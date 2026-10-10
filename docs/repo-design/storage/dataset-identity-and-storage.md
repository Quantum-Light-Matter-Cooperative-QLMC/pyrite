# Dataset identity, outputs, and storage

PyRITE separates a **dataset identity** (the complete resolved run) from a **case content key** (one reusable computation). This allows exact case reuse without allowing incompatible runs to resume into each other.

## Dataset identity

`cxr.dataset-identity.v1` contains the material, fidelity, optional variant, resolved settings, sweep, crystallography, and any non-default catalog-profile name. Canonical JSON serialization is hashed with SHA-256. The full payload and `parameter_sha256` are written to the component manifest as `dataset_identity`; the directory stem carries a 12-character digest for human navigation, not as the sole provenance record.

The standard, unmodified `full` run keeps the compatibility stem `<material>`. Other fidelities, catalog profiles, or explicit overrides use an identity-qualified stem `<material>@<label>-<12-char-digest>`, where the label is the explicit variant, non-standard catalog profile, or fidelity. Older `<material>--<fidelity>-<digest>` stems remain readable. The historical quick-run stem is retained but still records identity metadata.

## Case content keys and CAS

`cxr.case-content-key.v1` hashes the resolved fields that determine stored line and bremsstrahlung arrays. Presentation labels, detector solid angle, source cadence/charge, analytic mosaic broadening, and performance chunk sizes are excluded because they do not change those arrays. Profile names are labels: two profiles that resolve an identical case share the same key.

Case arrays live in a content-addressed store (CAS). Component checkpoints refer to those immutable blobs and retain per-dataset derived metadata. The cache can therefore reuse a case across datasets while recomputing requester- specific scale and reporting values.

Current array leaves use [HDF5 result schema version 2](result-schema.md). Its root records `identity_version = 1` for independent inspection, while the manifest remains authoritative for the full resolved dataset identity.

## Workspace layout

```text
<workspace>/
  pyrite-output/checkpoints/
    <stem>/
      line.h5
      brem.h5
      characteristic.h5
      meta.json and component metadata
    <material>/<first-two-case-key-chars>/<case-key>.h5
    archive/<label>/<stem>/...
  energy-grid-artifacts/<first-two-hash-chars>/<sha256>.json
  pyrite-output/performance/<profile>/...
  pyrite-output/observations/<stem>/...
  pyrite-output/trajectories/<stem>/...
  pyrite-output/results/
  pyrite-output/figures/
  pyrite-output/cache/  # regenerable; safe to delete
```

The exact internal files are implementation details; use `pyrite checkpoint` and the `pyrite material energy-grid` / `pyrite-dev energy-grid` commands rather than editing them. A command's `--checkpoint-dir` overrides the workspace checkpoint root for that invocation.

## Lifecycle and safety

- Active datasets live beneath the effective checkpoint root and are exposed by checkpoint-consuming workflows such as the analysis app. `checkpoint list` inventories only the long-term archive shelf.
- `checkpoint archive` and `restore` copy complete component directories, including provenance.
- `checkpoint merge` rejects identity-bearing datasets with different resolved hashes.
- `checkpoint gc` retains only datasets exactly matching current profile resolution; preview is the default.
- `checkpoint rm` deletes selected datasets and blobs kept reachable only by them; preview is the default.
- `pyrite-dev energy-grid verify` checks immutable object bytes and profile references; `pyrite-dev energy-grid gc` reclaims unreachable objects after its grace period.

Legacy checkpoints remain readable, but missing identity cannot be reconstructed retroactively. Treat them as provenance-incomplete. For case-store mechanics, see [Checkpoint case store](checkpoint-case-store.md); for user operations, see [Working with results](../../guides/working-with-results.md).
