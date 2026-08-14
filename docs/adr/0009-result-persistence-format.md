# 0009 — Result persistence format

- **Status:** Accepted
- **Date:** 2026-08-13
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
users who need one to interoperability instead. That promise needs a format.

## Decision

**Array payloads are stored as HDF5**, under a versioned schema documented at
{doc}`../repo-design/storage/result-schema`.

**`mcpl` and `h5py` are required dependencies**, not extras.

**MCPL is the interchange export** for emitted photons, written through the
official `libmcpl` C writer.

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

### `mcpl`, the full meta-package, required

`mcpl-core` ships prebuilt wheels for macOS x86-64 and arm64, manylinux x86-64
and aarch64, musllinux, and Windows amd64 and arm64. `mcpl-python` is pure
Python needing only `numpy>=1.22`. Both are Apache-2.0.

There is no build risk for an extra to hedge against. And a conditionally
available escape hatch would weaken the non-goal it exists to justify: ADR-0008
answers "can another code model our detector?" with "export MCPL", and that
answer cannot be contingent on how the user installed PyRITE.

Do not split it into `mcpl-python` alone to save a wheel. `mcpl-python` is a
*reader*; the writer lives in `libmcpl`, which only `mcpl-core` ships.

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
- Compression moves inside the container. HDF5 per-dataset filters replace the
  whole-stream zstd frame, so the artifact is self-describing at the cost of a
  measured write-throughput change recorded in the schema document.
- The validation ledger's evidence files become independently inspectable, which
  is the practical motivation.
- Control-plane state is untouched. Job records and remote lifecycle state are
  not pickled today (the campaign lock is already JSON), so there is nothing to
  convert and no decision owed here.
- MCPL export is a lossy *projection*, not a serialisation of the result. It
  carries what MCPL's particle record can carry. The schema document names every
  field that does not survive.
- Reversing this requires a superseding ADR. The pickle reader would remain
  regardless — it is permanent by the decision above, independent of what new
  writes use.
