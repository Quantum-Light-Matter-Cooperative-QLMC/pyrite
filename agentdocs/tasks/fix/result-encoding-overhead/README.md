# Result encoding and transfer overhead

Branch: `fix/result-encoding-overhead`

## Problem and evidence

`pyrite remote pull --profile hopg_hbn` took about 26 minutes on 2026-08-14 for
the HOPG and h-BN stores (roughly 540 MB + 530 MB for HOPG, roughly 250 MB per
component for h-BN). The cause is the version-1 HDF5 result encoding adopted in
[ADR-0009](../../../../docs/adr/0009-result-persistence-format.md), not the SSH
transport and not the chunk-handoff fixes in `c06f91e`.

Measured on `checkpoints/hopg@promising_coh-c33239a17398/brem.pkl` (6.7 MB on
disk, 21 MB raw pickle, 50,383 tree nodes, of which **2,916 are arrays**
totalling 24.7 MB; median array 8 KB, largest 16 KB):

| Encoding | Size | Write | Read | Container zstd-3 |
| --- | --- | --- | --- | --- |
| Legacy zstd-pickle | 6.6 MB | 0.03 s | 0.03 s | — |
| Current schema v1 | 42.5 MB | 6.9 s | 6.5 s | 6.6 MB in 0.04 s |
| Sketch: scalars as attributes | 25.9 MB | 2.09 s | 0.55 s | 5.8 MB in 0.03 s |
| Sketch: attributes, no array filter | 30.7 MB | 1.38 s | — | 7.0 MB in 0.04 s |
| **Sketch: record table, gzip blobs** | 7.97 MB | 0.26 s | 0.115 s | **5.32 MB in 0.013 s** |
| **Sketch: record table, unfiltered blobs** | **7.70 MB** | **0.105 s** | **0.073 s** | 6.52 MB in 0.027 s |

So the current encoding costs **6.5x the bytes and roughly 200x the CPU** of the
format it replaced, in both directions. The record-table layout is **66x faster
to write, 89x faster to read, and 5.5x smaller** than the version-1 encoding it
replaces, and framed for transfer it is smaller than the legacy pickle ever was.

Two independent causes, both confirmed by measurement:

1. **One HDF5 object per Python leaf.** `_write_node`
   (`src/pyrite/checkpoints/_checkpoint_io.py:48`) creates a dataset for every
   `str`, `int`, `float`, `bool`, and `None`. In this store 41,958 of 50,383
   nodes are such leaves (20,088 `str`, 9,234 `int`, 7,290 `float`, 3,888
   `None`, 972 `float64`, 486 `bool`) — mostly mapping keys and per-record
   scalar metadata —
   and each costs roughly 840 B of HDF5 object-header metadata and roughly
   125 us to create. About 36 MB of the 42.5 MB file is metadata describing a
   few hundred KB of scalars. A synthetic check isolates the per-object cost at
   roughly 80 us / 1.0 KB bare, and roughly 135 us / 2.7 KB once the gzip filter
   pipeline is attached to a small dataset.
2. **No whole-container codec on the wire.** That metadata is highly redundant
   and compresses roughly 6.4:1, but `pyrite slim -o -` streams it raw. Per-array
   gzip cannot see cross-record redundancy that whole-stream zstd exploited:
   24.7 MB of arrays gzip to roughly 20 MB, while the entire legacy artifact was
   6.6 MB.

The store's shape supports a table directly. The reference component holds
**486 records** (162 configurations x 3 beam energies) under exactly **one**
record keyset (`E_grid`, `E_grid_brem`, `brem`, `brem_wide`, `case`, `scale`),
and `case` flattens to 32 further columns. That is 38 columns and 486 rows —
roughly 1,000 HDF5 objects instead of 50,383.

Two further findings from the table sketch:

- **Array content is 3.5x duplicated.** The 2,916 arrays hold 24.7 MB but only
  **7.0 MB across 978 distinct blobs**: `E_grid` and `E_grid_brem` are identical
  across all 162 configurations at a given beam energy, and `case.E_grid`
  repeats the record's own grid. A content-addressed blob pool with an index
  column removes that redundancy without any semantic assumption — it is the
  same content addressing the CAS layer already uses.
- **Per-array gzip is a pure loss on this data.** Filtered blobs are both
  *larger* (7.97 MB vs 7.70 MB) and 2.4x slower to write than unfiltered ones,
  because deduplicated `float64` spectra have high-entropy mantissas while the
  chunk and filter-pipeline metadata is not free. Compression belongs in the
  container frame, not per dataset.

Non-causes, checked and excluded:

- Compression level. Level 1 and level 9 both produce 42.5 MB in 6.9 s.
- The spooled `dump_stream` staging (`_checkpoint_io.py:165`). Writing through
  the fileobj driver costs 6.89 s against 6.27 s for the native path driver,
  roughly 10%. It does defeat the encode/transfer overlap claimed at
  `src/pyrite/remote/lifecycle.py:1510-1517` and documented at
  `docs/repo-design/storage/result-schema.md:79-81`; that claim is wrong today
  and must be corrected either way.

This is the same root cause as the write-side symptoms already recorded in
[`fix/chunked-checkpoint-lifecycle`](../chunked-checkpoint-lifecycle/README.md):
HOPG serialized `brem.pkl` in about 95 s (5.7 MB/s) and decoded a 1.07 GB
checkpoint in about 155 s (6.9 MB/s), both consistent with the per-node costs
above.

Reproduction: load a component store through `_checkpoint_io.load`, re-encode it
with `_checkpoint_io.dump`, and compare size and wall time against
`compression.zstd.compress(pickle.dumps(obj, 5), 3)`. Node and array census by
recursive walk over the loaded object.

## Scope

Own the leaf encoding and the pull wire format:

- schema version 2 of `pyrite.result`, and the reader that must keep accepting
  version 1 and all three legacy pickle generations;
- the transfer framing used by `pyrite slim -o -` and `remote pull`;
- the schema document's promised — and still missing — throughput record.

Preserve:

- ADR-0009's decision in full. The artifact stays HDF5, **pickle-free**,
  openable with `h5py.File` and readable with no PyRITE import and no code
  execution. A pickled or otherwise opaque skeleton blob is rejected on those
  grounds even though it measured 21 MB / 0.60 s / 0.35 s.
- every logical shape version 1 preserves: mapping order, non-string mapping
  keys, absent-vs-null entries, list/tuple kind, array dtype and shape,
  NumPy-scalar dtype, exact float bits;
- byte-stability of a given payload across paths and write times
  (`tests/checkpoint/test_io.py:31`);
- atomic write-then-rename, CAS blob semantics, and dataset identity;
- readability of every artifact already written — there are version-1 stores on
  the box and in `checkpoints/` now, and no migration script is permitted.

Non-goals:

- renaming `.pkl` to `.h5` (separate `TODO.md` ergonomics item);
- shard lifecycle, consolidation policy, resume fast paths, or progress
  reporting — owned by `fix/chunked-checkpoint-lifecycle`;
- physics, RNG, or spectrum changes;
- a new interchange format (ADR-0009 defers it).

Boundary with `fix/chunked-checkpoint-lifecycle`: that task owns **when** the
store reads and writes; this task owns **what bytes** a read or write costs.
Both touch `_checkpoint_store` and both cite the same 2026-08-14 workload, so
whichever lands second rebases and re-measures. Its README explicitly defers
"HDF5 schema redesign" and forbids "tuning HDF5 compression without a
representative A/B measurement"; the table above is that measurement.

## Implementation path

### 1. Baseline and guardrails

- Add a size/time regression test over a synthetic store with a realistic node
  census (thousands of small arrays, tens of thousands of scalar leaves) that
  fails if bytes-per-node or seconds-per-node regress past a recorded bound.
- Extend `tests/checkpoint/test_io.py` with a version-1 fixture artifact
  committed as a small binary, proving forward reads never break.
- Record the current numbers as the baseline in the schema document.

### 2. Schema version 2: a record table, not a node per scalar

The payload is columnar, so encode it columnar. Decided against attribute
packing on the measurements above.

- Detect a **record set**: a mapping whose nesting bottoms out in leaf mappings
  sharing one key set. Encode it as a `record-table` group: a key column per
  nesting level (`configuration`, `E0_keV`), then one dataset per flattened
  record column. Nested mappings such as `case` flatten to dotted or grouped
  column paths and keep a structure marker so the reader rebuilds the exact
  nesting.
- Store arrays in a **content-addressed blob pool** with an `array-ref` index
  column per array field. Readers materialize a fresh array per reference —
  deduplication must never alias two record fields to one mutable object.
- Encode sequence-valued columns (`composition`, `hkl_list`, `beam_uvw`, the
  `E_grid_brem` tuple) with a typed rule: a 2D dataset for uniform-length
  numeric tuples, a ragged values-plus-offsets pair otherwise. The sketch used
  a JSON string column as a placeholder and its numbers include that cost; JSON
  survives into the final schema only if a typed encoding measurably loses.
- Represent nulls with an explicit per-column boolean mask, not an in-band
  sentinel. The sketch needed a one-byte opaque sentinel because HDF5
  variable-length strings reject embedded NULs; a mask distinguishes absent from
  null without inventing a magic value.
- Write blobs **unfiltered**. Per-array gzip is measurably counterproductive
  here (see above); the container frame in step 3 does the compression.
- Keep a **typed-tree fallback** for payloads that are not uniform record sets —
  CAS runner blobs, `anchor_figures` caches, `archive.union_checkpoint` output.
  Use the measured attribute-packed form for that path, so the fallback is
  25.9 MB / 2.09 s rather than the version-1 42.5 MB / 6.9 s.
- Bump `SCHEMA_VERSION` to 2 and dispatch reads on `schema_version`, retaining
  the version-1 reader permanently alongside the pickle readers.
- Keep it self-describing: `kind` attributes stay, and the `visit(print)`
  inspection recipe must still yield a legible tree — a column-per-field table
  is *more* legible to `h5dump` than the numbered node tree it replaces.

Target from the sketch: 7.7 MB, 0.105 s write, 0.073 s read for the reference
store — 66x faster write, 89x faster read, 5.5x smaller than version 1.

### 3. Whole-container codec for transfer

- Frame the `pyrite slim -o -` stream with zstd so the box sends 5-7 MB where it
  now sends 42.5 MB per reference-store equivalent. Python 3.14 has
  `compression.zstd` in the standard library, so this needs no binary on the
  box; `ssh -C` is the zero-code alternative and should be measured against it.
- Teach the receiving side to unframe. `_checkpoint_io.load` already sniffs the
  zstd magic but then assumes a **pickle** payload; a zstd-framed HDF5 artifact
  would be misread today. Fix the sniff to decompress and then re-dispatch on
  the inner magic, which also makes framed artifacts loadable by hand.
- Keep on-disk artifacts unframed so `h5py.File` still opens them directly.
  Framing is a transfer concern only; do not let it leak into the CAS layout.
- Report the framed and unframed sizes in the existing pull throughput line
  (`lifecycle.py:1525-1531`) so a future slow pull stays attributable.

### 4. Fix the overlap claim

- Either restore genuine encode/transfer overlap, or correct
  `lifecycle.py:1510-1517` and `result-schema.md:79-81` to state that transfer
  begins after encoding. Do not leave a comment that asserts an overlap the
  spooled staging prevents.

### 5. Measured closure

- Fill in the write-throughput record ADR-0009 promised
  (`docs/adr/0009-result-persistence-format.md:104-106`) and the schema document
  never received.
- Update `docs/repo-design/storage/result-schema.md` for version 2: the typed
  tree table, the compatibility section, and the inspection recipe.
- Re-measure a real `pyrite remote pull --profile hopg_hbn` against the
  26-minute baseline. Do not run a heavy sweep locally.

## Decisions

- HDF5 stays. ADR-0009 is not reopened; this task fixes an implementation that
  spends the format badly, and no superseding ADR is required.
- Pickle-free is non-negotiable, so the cheapest prototype is rejected on
  principle rather than on measurement.
- Version 1 is read forever. There is no flag day and no migration script;
  stores upgrade opportunistically on the next save, exactly as ADR-0009
  specifies for the pickle generations.
- Both fixes land, in this order. The codec alone would restore pull wall time
  while leaving the 95 s serialization stalls in place; the layout alone leaves
  recoverable bytes on the wire.
- **A record table, not attribute packing.** User decision on the measurements
  above: attributes cut version-1 cost by only 3.3x on write and left 25.9 MB,
  which is still far off the format it replaced. The table is 66x/89x/5.5x and
  is the idiomatic HDF5 encoding for a `{configuration: {E0_keV: record}}`
  store. Attribute packing survives only as the non-table fallback path.
- **Content-addressed array blobs.** Deduplication is what takes the array
  payload from 24.7 MB to 7.0 MB; it is not an optional optimization, it is why
  the table beats the legacy pickle on the wire.
- **Compression lives in the container frame, not the dataset filter.**
  Unfiltered blobs are smaller *and* 2.4x faster to write than gzip-filtered
  ones once duplicates are gone. This reverses the earlier draft of this
  document and the ADR's "compression moves inside the container" consequence;
  the ADR's decision (HDF5, self-describing, pickle-free) is untouched, only its
  filter tactic. Frame choice on the wire is a measured trade: filtered blobs
  plus zstd reach 5.32 MB for 0.14 s more CPU than unfiltered plus zstd at
  6.52 MB. Pick by link rate; default to the cheaper CPU.

## Open questions for implementation owner

- Where the record-set detector draws its boundary, and what it does with a
  *nearly* uniform set — one record carrying an extra optional key such as
  `spec_coherent` must not force the whole store onto the fallback path. A
  per-column presence mask handles it; confirm against a store that mixes
  coherent and non-coherent records.
- Whether the blob pool is per-artifact or reaches the existing CAS layer. This
  task assumes per-artifact; sharing across artifacts is a store-layout change
  that belongs to `fix/chunked-checkpoint-lifecycle`, not here.
- Whether one compound dataset per record set beats one dataset per column. The
  sketch measured column-per-dataset; a compound row type may read faster for
  whole-record access but is worse for column scans and for `h5dump` legibility.
- The typed encoding for sequence columns, per step 2 — this is the one place
  the sketch's numbers rest on a placeholder.
- Whether the transfer frame is a `slim` flag or unconditional, and how an older
  local `pyrite` reacts to a framed stream from a newer box after `sync_code`.
  Version skew across the SSH boundary is real: pick a rule that fails loudly.
- Whether byte-stability must hold across schema versions or only within one.
- Whether `archive.union_checkpoint` and `runs.run.cached_material_analysis`
  need any change beyond inheriting the new writer.

## Delegation slices

1. **Schema version 2 encode/decode** — owner: `implement-task` +
   `performance` + `regression-testing`. Not `one-shot`: the attribute-versus-
   record-set boundary, the sequence-column encoding, and the fallback-path
   split are open. Covers `_checkpoint_io`, `tests/checkpoint/test_io.py`, and
   the version-1 fixture.
2. **Transfer codec and pull path** — owner: `implement-task` +
   `remote-gpu-jobs`. Depends on slice 1 only for the corrected magic sniff;
   self-contained enough for Serena `one-shot` once the framing rule and skew
   behaviour are fixed. Covers `slim.py`, `remote/lifecycle.py`,
   `remote/transport.py`, `tests/remote/test_remote.py`,
   `tests/checkpoint/test_slim.py`.
3. **Documented measurement and real-pull closure** — owner: `lead-task` +
   `performance` + `documentation-maintenance` + `remote-gpu-jobs`. Integrates
   both slices, fills the ADR-promised throughput record, and re-measures the
   real pull.

No slice may push, edit `TODO.md`, retire task records, or delegate further
unless the dispatcher explicitly grants that authority.

## Acceptance checks

- Reference store `hopg@promising_coh-c33239a17398/brem.pkl` re-encoded:
  - at most 9 MB, at most 0.30 s write, at most 0.20 s read;
  - byte-identical output across two writes to different paths;
  - exact round-trip of mapping order, non-string keys, nulls, list/tuple kind,
    dtypes, and float bits;
  - no two record fields aliased to one array object after load.
- A store mixing coherent and non-coherent records (differing optional keys)
  still encodes as one table, not as the fallback tree.
- Every version-1 and legacy-pickle fixture still loads, and no artifact is
  rewritten merely by being read.
- A version-2 artifact opens with `h5py.File`, reports
  `schema_version == 2`, and `visit(print)` yields a legible tree with no
  opaque blob standing in for the structure.
- `pyrite slim -o -` transfers at most 7 MB for the reference store, and the
  pull throughput line reports both framed and unframed sizes.
- A real `pyrite remote pull --profile hopg_hbn` completes in a small fraction
  of the 26-minute baseline, with the measured before/after recorded in
  `docs/repo-design/storage/result-schema.md`.
- `uv run pyrite-dev verify` passes.
