# Distributed worker artifacts

`pyrite.storage.CaseArtifactStore` publishes immutable raw worker outputs;
`CaseArtifactRef` identifies a published artifact. The existing
`pyrite.execution` and `pyrite.execution.artifacts` imports remain compatibility
paths. Scheduling belongs to the [execution profiles](cluster-execution.md);
only the coordinator owns authoritative checkpoint writes.

A worker executes the existing PyRITE numerical code and publishes its
**raw** `run_case` output to a shared filesystem:

```python
ref = artifact_store.publish(
    material=case["crystal"], content_key=content_key_fn(case), output=run_case(case)
)
# Send only `ref` (a small dataclass) back to the coordinator.
```

The coordinator supplies that same store and content-key function to `run_sweep`:

```python
run_sweep(
    cases,
    results,
    executor=my_executor,
    artifact_store=artifact_store,
    content_key_fn=content_key_fn,
    checkpoint_path=checkpoint_path,
)
```

An executor must invoke `callback(index, case, ref)` in the **coordinating
process**; the callback validates the stored artifact's file digest, format,
material and independently computed physics key before handing the arrays into
the existing `store_result` / CAS / checkpoint lifecycle. Workers never own
checkpoint manifests. Use the existing `case_content_key` with any required
external-table markers, not a hand-rolled hash of scheduler metadata.

Publication creates a temporary `.tmp` file, writes a complete HDF5 payload,
fsyncs it, and atomically links it into a per-case inbox path. Publication
is complete only after a separate atomic checksum receipt is committed. A crash
between those steps leaves an uncommitted candidate that a retry can safely
finish; competing workers never overwrite a published file.
On restart with `resume=True` and `cache_read=True`, a coordinator can recover
published work not yet committed to a checkpoint, before dispatching remaining
cases. Corrupt completed artifacts raise rather than being treated as cache hits.

## Identities and limits

The store requires a shared POSIX-like filesystem supporting hard links.
`case_content_key` identifies *inputs*; the SHA-256 on the reference identifies
the published *bytes*. These are not interchangeable. Do not use the inbox as
a substitute for authoritative checkpoint persistence.

Filesystem space reclamation, standalone staging garbage collection,
coordinator locking across multiple simultaneous campaigns and object-store
publication are outside this store's contract. The SLURM and Dask executors
provide parallel scheduling separately; their restrictions are listed in the
[cluster execution reference](cluster-execution.md#validation-and-limitations).
Directional work units use the executors' direction-aware artifact handling;
a raw case artifact is not itself an analysis-ready directional result.

For persisted observations, captures and result values, see the
[result/storage contracts](../repo-design/result-storage-contracts.md).
The publication and integrity implementation is
{source}`src/pyrite/execution/artifacts.py::CaseArtifactStore`.
