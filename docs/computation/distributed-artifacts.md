# Worker-safe distributed case artifacts (issue #681, milestone 2)

This opt-in slice introduces `pyrite.execution.CaseArtifactStore` and
`CaseArtifactRef`. It intentionally does **not** introduce a remote scheduler.

The remote worker executes the existing PyRITE numerical code and publishes its
**raw** `run_case` output to a shared filesystem:

```python
ref = artifact_store.publish(material=case["crystal"],
                             content_key=content_key_fn(case),
                             output=run_case(case))
# Send only `ref` (a small dataclass) back to the coordinator.
```

The coordinator supplies that same store and content-key function to `run_sweep`:

```python
run_sweep(
    cases, results, executor=my_executor, artifact_store=artifact_store,
    content_key_fn=content_key_fn, checkpoint_path=checkpoint_path,
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

**Limits / next steps:** Requires shared POSIX-like filesystem supporting hard
links. Filesystem space reclamation, standalone staging garbage collection,
coordinator locking across multiple simultaneous campaigns, object-store
publication, directional-observation replay, and parallel scheduling are not
implemented in this slice. `case_content_key` identifies *inputs*; the SHA-256
on the reference identifies the published *bytes*. These are not interchangeable.
Do not use the inbox as a substitute for authoritative checkpoint persistence.
