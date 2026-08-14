# 0009 — Result persistence format

- **Status:** Accepted
- **Date:** 2026-08-13
- **Revised:** 2026-08-14 — defer interchange format selection
- **Context source:** core architecture RFC, Change 6

## Context

PyRITE's results are pickles. Component stores (`checkpoints/<stem>/{line,brem}.pkl`),
per-config crash shards, content-addressed case blobs, and `slim` transfer
artifacts are all the same thing: a `pickle` stream behind a compression frame,
written and read by one module, `checkpoints/_checkpoint_io`.

The store *design* is sound. Content addressing, sharding, atomic
write-then-rename, and the campaign lock all work and are not in question. The
problem is the leaf encoding.

Pickle is a poor archival format for results that back scientific claims:

- It is version-fragile. A pickle is a program for a particular object graph;
  Python and NumPy releases have broken pickles before and will again.
- It is unsafe to accept from a third party. Loading one executes code, so a
  checkpoint received from a collaborator cannot be opened without trusting them
  completely.
- It is unreadable by anything that is not Python with PyRITE importable. A
  reviewer with the file and no repository cannot see what is in it.

This repository maintains a physics validation ledger whose evidence lives in
those files. Evidence that only one program can read, and only while that
program's dependencies stay pinned, is weak evidence.

Separately, ADR-0008 declines to build a general geometry system and routes
downstream transport needs to interoperability. That strategy eventually needs
a source contract and an adapter, but neither is a result-persistence concern.

## Decision

**Array payloads are stored as HDF5**, under a versioned schema documented at
{doc}`../repo-design/storage/result-schema`.

**`h5py` is a required dependency.** No interchange-format dependency is added
by this decision.

### HDF5, not Zarr

Zarr's advantage is many concurrent writers into one large array. That case does
not arise here. PyRITE writes many small independent per-case artifacts — one
blob per content key, one shard per finished config — and never has two writers
inside one array.

Against that, Zarr's directory-of-chunks layout fights the store's
write-then-rename atomicity. A single `.h5` file renames atomically and maps
one-to-one onto a CAS blob; a chunk directory does not rename atomically, and
would force either a lock or a tree-copy discipline that the store does not have
and this change is not permitted to add.

The decision is atomicity and blob semantics. Reader popularity is not the
argument — though HDF5 also happens to be what `h5py`, MATLAB, Julia, IDL, and
`h5dump` already read.

### Interchange format deferred

The RFC originally selected required MCPL export. Inspection during
implementation showed that the stored result is an energy-binned spectral
tally, not an emitted-photon list. It contains no joint position, direction,
energy, time, polarization, and statistically normalized weight record. Writing
MCPL from it would require a new source estimator or a documented lossy
projection; neither is a file-format operation.

Interchange work is therefore deferred and ordered explicitly:

1. Identify a concrete downstream consumer and its transport boundary.
2. Define a format-neutral `PhotonSource` contract and physical closure tests.
3. Select and test the adapter that matches that workflow.

MCPL remains a strong candidate when the consumer is Geant4, OpenMC,
MCNP/PHITS, or McStas/McXtrace. It is not required until such a workflow and
source-state contract justify it.

### The filename does not change

Files keep their `.pkl` names. 136 literal `.pkl` references across 22 modules
encode the CAS and component layout, which this change holds fixed;
`discover()` globs `*.pkl` and `checkpoint_exists` stats `line.pkl`.

`_checkpoint_io.load` has never trusted the extension — it sniffs magic bytes,
and already dispatches across three generations of stored bytes (plain pickle,
gzip, zstd). HDF5 is the fourth, recognised by its own signature. The extension
is a path token, not a format claim.

### Legacy files load forever

There is no flag day and no bulk migration script. Every `.pkl` dataset ever
written stays loadable, indefinitely, by the same `load` call. Migration is
opportunistic: a store is rewritten as HDF5 the next time something saves it,
which is exactly what the store already does when it meets the legacy flat
layout.

## Consequences

- A result file can be opened with `h5py.File(path)` and read with no PyRITE
  import, no `sys.path` arrangement, and no code execution.
- Compression leaves the stored artifact entirely. This ADR originally expected
  HDF5 per-dataset filters to replace the whole-stream zstd frame; measurement
  refuted that. Deflate costs about 2.5x the write time and, once array content
  is deduplicated, saves nothing a whole-container frame does not save better,
  so artifacts are unfiltered and self-describing. The zstd frame survives as a
  transfer codec for `slim -o -` and is never stored. Numbers are in the schema
  document.
- The validation ledger's evidence files become independently inspectable, which
  is the practical motivation.
- Control-plane state is untouched. Job records and remote lifecycle state are
  not pickled today (the campaign lock is already JSON), so there is nothing to
  convert and no decision owed here.
- Result persistence does not fabricate particle phase space. A future
  interchange adapter is owned by the format-neutral photon-source boundary,
  not by this HDF5 schema.
- Reversing this requires a superseding ADR. The pickle reader would remain
  regardless — it is permanent by the decision above, independent of what new
  writes use.
