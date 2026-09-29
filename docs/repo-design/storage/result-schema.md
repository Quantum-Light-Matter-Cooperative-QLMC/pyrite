# Result persistence schema

PyRITE result artifacts use HDF5 schema `pyrite.result`, version 2. Current component, shard, and CAS paths end in `.h5`. Historical `.pkl` paths remain readable; readers identify the encoding from its signature rather than its suffix. Open a current artifact directly with `h5py.File("line.h5")` or any HDF5 reader.

Version 1 is read forever and never written again. See [Compatibility and writes](#compatibility-and-writes).

## Container header

The HDF5 root has three required attributes and one contract marker:

| Attribute | Type | Current value | Meaning |
| --- | --- | --- | --- |
| `schema` | UTF-8 string | `pyrite.result` | Format discriminator |
| `schema_version` | integer | `2` | Container/tree encoding version |
| `identity_version` | integer | `1` | Dataset-identity normalization version |
| `emission_components` | UTF-8 string | `separate` | Emission arrays exclude one another; absent on older artifacts |

A container without `emission_components`, and every pickle, predates the separate-component contract: its `spec` and `spec_coherent` include any co-located `spec_characteristic`. The reader subtracts that component once, so loaded records always follow the current contract. The marker is on the container rather than on records so slim, dataset-merge, and basket key projections cannot drop it.

The root group `value` contains the encoded payload; the root group `blobs`, when present, is the array pool. Readers must reject an unknown `schema` or `schema_version`; identity compatibility remains governed by the checkpoint manifest and dataset-identity rules.

## Why version 2 exists

Version 1 gave every Python leaf its own HDF5 object. An object header costs about 840 B and about 125 µs, so a reference component store — 50,383 nodes, of which only 2,916 were arrays — spent roughly 36 MB of its 42.5 MB on metadata describing a few hundred kB of scalars, and `pyrite remote pull` took 26 minutes for two ~500–600 MB stores. Measured on that store:

| Encoding | Bytes | Write | Read | Framed for transfer |
| --- | --- | --- | --- | --- |
| Version 1 typed tree | 42.5 MB | 6.9 s | 6.5 s | 6.6 MB |
| Version 2, gzip+shuffle blobs | 7.97 MB | 0.26 s | 0.115 s | 5.32 MB |
| Version 2, unfiltered | 7.70 MB | 0.105 s | 0.073 s | 6.52 MB |
| Version 2, shuffle only | 9.72 MB | 0.149 s | 0.085 s | 4.67 MB |

Three results set the design. Array content is 3.5× duplicated (24.7 MB of arrays over 7.0 MB of distinct content) because `E_grid` and `E_grid_brem` repeat across every configuration at a given energy. Deflate costs roughly 2.5× the write time and, once that duplication is gone, saves nothing a whole-container frame does not save better. What deflate's `shuffle` companion filter does pay for is compressibility, so it is applied only in front of that frame.

Stored artifacts are therefore written **unfiltered**; the transfer artifact is written with `shuffle` and framed whole.

## Record tables

A component store has the shape `{configuration: {E0_keV: record}}` — many rows over one key set — so it encodes columnar rather than as a tree.

A mapping becomes a `record-table` when peeling nested mapping levels reaches a uniform record layer of at least `MIN_TABLE_ROWS` (4) rows whose keys overlap by at least `MIN_KEY_DENSITY` (0.5) of the key union. Anything else — CAS runner blobs, analysis caches, small payloads — stays a typed tree.

A flat list or tuple of records — a cross-material comparison cache holds thousands of `{"E0_keV": …, "line_eV": …, "quality": …}` dicts — becomes a `record-list` under the same rule: at least `MIN_TABLE_ROWS` elements, every element a non-empty mapping, and key density at least `MIN_KEY_DENSITY`. It is a record table without key columns: the group carries `container` (`list` or `tuple`) and one `records` column set. Short, sparse, mixed, or non-mapping sequences stay a typed-tree `list`/`tuple`. A 6000-record comparison cache went from about 14 MB and seconds per dump or load as one group per record to about 0.2 MB and tens of milliseconds.

| Object | Contents |
| --- | --- |
| `keys/00`, `keys/01`, … | one column per nesting level above the record layer |
| `records` | a `column-set` over the record mappings |
| `records/cols/<name>` | one `column` per record key |
| `records/layout*` | pooled per-row key order and presence |

`layout` holds one pooled-signature index per row; `layout_values` and `layout_offsets` hold the distinct signatures. A uniform record set costs one signature, and an optional key such as `spec_coherent` adds one more — not a presence mask per column. A column present in only some rows is `dense = False` and carries its own `rows` index.

Column encodings, chosen by content:

| `enc` | Payload |
| --- | --- |
| `null` | every row null; no data |
| `scalar` | one `values` dataset of a single tag's dtype (UTF-8 for `str`) |
| `array-ref` | `refs` into the blob pool, `-1` for null |
| `column-set` | nested record columns for mapping-valued keys such as `case` |
| `seq-fixed` | equal-length numeric sequences as one 2D block |
| `seq-ragged` | unequal-length numeric sequences as `values` plus `offsets` |
| `tree-pool` | content-addressed typed trees plus `refs`, for anything else |

Nullable columns carry a `nulls` boolean dataset alongside their values.

## Blob pool

Numeric arrays inside a record table are stored once in `/blobs`, keyed by a 16-byte BLAKE2b digest over a type-tagged serialization, and referenced by index. **Readers materialize a fresh array per reference**, so deduplication is never observable as aliasing — callers mutate spectra in place.

## Typed tree encoding

Payloads that are not record sets, and the record table's own `tree-pool` entries, use the typed tree. Every node has a UTF-8 `kind` attribute. The encoding preserves mapping order, non-string mapping keys, absent mapping entries, explicit nulls, list/tuple kind, array dtype, and NumPy-scalar dtype.

| `kind` | HDF5 object | Payload |
| --- | --- | --- |
| `mapping` | group | one child or attribute per entry; see below |
| `list`, `tuple` | group | entries named `00000000`, … |
| `packed-sequence` | dataset | a uniform-tag numeric sequence as one array |
| `record-table` | group | see [Record tables](#record-tables) |
| `record-list` | group | a uniform list/tuple of records: `container` attribute plus a `records` column set; see [Record tables](#record-tables) |
| `array` | dataset | native NumPy numeric/boolean/byte array with exact shape and dtype |
| `unicode-array` | dataset | variable-width UTF-8 values plus `numpy_dtype` attribute |
| `scalar` | group | `tag` plus a `v` attribute; `null` carries no value |
| `bytes` | `uint8[N]` dataset | uninterpreted bytes |

Scalar entries do not get their own object: a group stores them in attributes named `v:<name>`, with a parallel `kinds` attribute naming each entry's tag (`node` for entries that are children). This is what removes version 1's per-leaf object cost. Mapping keys are inline in a `keys` attribute when all of them are strings, and otherwise written as a `_keys` sequence node.

Child and attribute names are the literal string key when it is non-empty, unique, and free of `/`, `.`, and a leading `_`, so `h5dump` shows real field names; anything else takes an `_00000000` index. Keys that themselves begin with `_` never take the literal form, so the two namespaces cannot collide.

Object-dtype arrays and arbitrary Python objects are rejected. Nothing is unpickled on load.

## Result-store semantics

Component stores have the logical shape `{configuration: {E0_keV: record}}`. CAS blobs contain the runner output mapping for one case. The encoding above is the normative physical form; these tables define the scientific fields.

| Record field | Dtype | Shape | Unit / meaning |
| --- | --- | --- | --- |
| `E_grid` | `float64` | `[N]` | photon energy, eV |
| `spec` | `float64` | `[N]` | incoherent PXR/CBS line spectrum, photons/e/sr/eV; excludes characteristic radiation |
| `spec_coherent` | `float64` | `[N]` | coherent PXR/CBS companion spectrum; excludes characteristic radiation; key absent when not computed |
| `spec_characteristic` | `float64` | `[N]` | characteristic spectrum, photons/e/sr/eV; stored in `characteristic.h5`, co-located in CAS blobs and slim stores |
| `brem` | `float64` | `[N]` | bremsstrahlung interpolated to `E_grid`, photons/e/sr/eV |
| `E_grid_brem` | `float64` or null | `[M]` | wide bremsstrahlung photon-energy grid, eV |
| `brem_wide` | `float64` or null | `[M]` | wide bremsstrahlung spectrum, photons/e/sr/eV |
| `E_pk` | `float64` | scalar | line peak energy, eV |
| `fwhm` | `float64` | scalar | line full width at half maximum, eV |
| `eta` | `float64` | scalar | backscattered-electron fraction |
| `hit_frac` | `float64` | scalar | finite-footprint hit fraction; `NaN` for old records |
| `scale` | `float64` | scalar | conversion from per-e/sr to per-s/nA |
| `source_current_na` | `float64` | scalar | reporting source current, nA |
| `case` | mapping | — | version-1 resolved simulation case |

CAS runner mappings add `n_segments` (integer count), `crystal` (catalog key), `E0_keV` (incident electron energy in keV), and optional underscore-prefixed timing/backend diagnostics. Diagnostics are metadata, not scientific arrays.

## Compatibility and writes

Readers sniff the HDF5 eight-byte signature and then the container's `schema_version`, so version 1 and version 2 artifacts load side by side with no migration step and no flag day. If the signature is absent, readers permanently fall back to the previous zstd-compressed pickle, gzip-compressed pickle, and plain pickle readers; a zstd frame is disambiguated by the signature of what it decodes to. Loading does not rewrite. The next normal store save writes version 2 through the existing same-filesystem temporary file and `os.replace`.

`record-list` is an additive node kind within version 2, not a schema bump. Version 2 files written before it, which store a record list as a `list`/`tuple` group with one `mapping` child per record, still load unchanged. An older reader meeting `record-list` fails loudly with `unsupported result node kind`, the same failure a version bump would give, without rejecting files that do not use it.

`compresslevel` keeps its historical 1–22 interface for CLI compatibility but no longer affects a stored artifact. It selects the transfer frame's strength.

## Transfer frame

HDF5 requires random-access output, so `dump_stream` stages the container in a temporary directory and then compresses it onto the caller's pipe as a single zstd frame. `pyrite remote pull` loads that frame and deletes it immediately, so no framed artifact is ever stored.

The frame — not per-dataset filters — is what keeps a pull cheap: HDF5 object metadata is highly redundant, and `shuffle`-transposed arrays compress far better whole than one dataset at a time. Encoding does not overlap the transfer, because the staged container must be complete before its first byte ships.

## Independent inspection

This minimal reader requires h5py but no PyRITE import:

```python
import h5py

with h5py.File("checkpoints/hopg/line.h5", "r") as result:
    assert result.attrs["schema"] == "pyrite.result"
    print(result.attrs["schema_version"])
    result.visit(print)
```

External tools may read any array directly. Reconstructing the ordered Python-shaped mapping means following a record table's key columns and layout pool, or, in a typed tree, each group's `keys`, `names`, and `kinds` attributes.
