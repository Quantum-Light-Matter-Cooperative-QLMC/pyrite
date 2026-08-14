# Pickle inventory

Slice A for the result-format task. Classifies every `pickle` reference under
`src/` so slice D knows exactly which byte streams change encoding and which are
deliberately left alone.

`rg -l pickle src/` reports fifteen modules. Nine of them only *mention* pickles
in prose. Four modules make up the entire persistence surface, and all four
funnel through one codec.

## Classification

| Module | Lines | Class | What it actually does |
| --- | --- | --- | --- |
| `checkpoints/_checkpoint_io.py` | 33, 83, 99-104 | **Result array payload** | The only encoder/decoder. `dump`/`dump_stream`/`load`. Every other payload site calls it. |
| `checkpoints/_checkpoint_store.py` | via `_checkpoint_io` | **Result array payload** | Component stores `{line,brem}.pkl`, per-config shards `parts/<sha1>.pkl`, CAS blobs `<mat>/<2hex>/<key>.pkl`. |
| `checkpoints/slim.py` | 157 + `_checkpoint_io` | **Result array payload** | Writes `<stem>.slim.pkl` transfer artifacts and streams to stdout for `remote pull`. |
| `runs/run.py` | 246, 283, 715, + `_checkpoint_io` | **Mixed** | Checkpoint save/load is payload. `pickle.dumps` at 246 is *incidental* (stable hash input for a cache filename, never written). `.analysis-cache/*.pkl` is a **derived payload** (see below). |
| `checkpoints/archive.py` | 148, 279, 284, 319 | **Result array payload (indirect)** | Reads/merges/writes stores through `_checkpoint_io`; `_atomic_copy`/`_atomic_copytree` move opaque bytes and do not care about encoding. |
| `apps/analyze.py` | 34, 148 | **Reader, payload-adjacent** | Catches `pickle.UnpicklingError`/`gzip.BadGzipFile`/`zlib.error` to skip a checkpoint mid-`pull`. Must learn the HDF5 truncation exception too. |
| `remote/lifecycle.py` | 1465, + `_checkpoint_io.load` at 1533 | **Control plane** | Orchestration only. Pulls an artifact and re-encodes via the shared codec. Stores no pickled state of its own. |
| `montecarlo/runner/__init__.py` | 7, 47, 308, 1179 | **Control plane** | `multiprocessing` spawn IPC. Never touches disk. Out of scope permanently. |
| `devtools/package_smoke.py` | 105-123 | **Incidental** | Asserts a `cxr_mc`-era pickle of `Sweep` still resolves after the package rename. A compatibility *test*, not a store. |
| `cxr_mc/__init__.py` | 4 | **Incidental** | Module aliases that make those legacy pickles resolve. Load-bearing for slice E; no encoding of its own. |
| `checkpoints/campaign_lock.py` | 23 | **Incidental** | Lock metadata is `cxr.lock.json` / `<stem>.lock.json`. Already JSON. Only the *path* derivation mentions pickles. |
| `results/selection.py` | 293, 317 | **Incidental** | Docstrings. |
| `cli/commands/scan.py` | 79 | **Incidental** | `--dir` help text. |
| `cli/commands/blaze.py` | 121 | **Incidental** | `--dir` help text. |
| `cli/commands/slim.py` | 14-24 | **Incidental** | Command help text. |
| `remote/cli.py` | 196 | **Incidental** | Comment. |

Net: **one codec module to change**, three payload modules that inherit the
change for free, one reader to teach a new exception, and eleven modules that
need no edit at all.

## Payload shapes that must survive

### 1. The results store — `{name: {E0_keV: record}}`

Written by `results.store.store_result`. Record keys:

| Key | Type | Unit | Presence |
| --- | --- | --- | --- |
| `E_grid` | `float64[N]` | eV | Always |
| `spec` | `float64[N]` | photons/e/sr/eV on the line grid | Always |
| `brem` | `float64[N]` | photons/e/sr/eV, interpolated onto `E_grid` | Always |
| `E_grid_brem` | `float64[M] \| None` | eV, wide coarse grid | Always (may be `None`) |
| `brem_wide` | `float64[M] \| None` | photons/e/sr/eV on the wide grid | Always (may be `None`) |
| `E_pk` | `float` | eV, argmax of `spec` | Always |
| `fwhm` | `float` | eV | Always |
| `eta` | `float` | backscattered fraction, dimensionless | Always |
| `hit_frac` | `float` | dimensionless; `NaN` on pre-feature records | Always |
| `scale` | `float` | (per e per sr) to (per s per nA) | Always |
| `source_current_na` | `float` | nA | Always |
| `case` | `dict` | 43-key case payload | Always |
| `spec_coherent` | `float64[N]` | as `spec` | **Conditional** — absent, not `None`, for incoherent runs |

`spec_coherent`'s *absence* is semantically load-bearing: `analyze.emission_menu`,
the altair overlay, and `run.repair_line_spec` all gate on `key in record`. The
schema must distinguish "absent" from "present and null".

### 2. CAS blob — one transport `out` dict

Built at `montecarlo/runner/__init__.py:1138`. Same array keys as above plus
`n_segments` (`int`), `crystal` (`str`), `E0_keV` (`float`), and, under
`timed=True`, a set of underscore-prefixed diagnostics
(`_t_spectrum`, `_t_transport`, `_gpu_oom_retries`, `_allocator_*_mib`,
`_backend`, `_backend_vendor`, `_backend_device`, `_attempted_*_chunk`,
`_effective_*_chunk`). Mixed `int`/`float`/`str` scalars, all leaves.

### 3. Analysis cache — `{"checkpoint": <signature tuple>, "value": <analysis>}`

`runs/run.py:290`. The signature is a tuple of `(path_str, mtime_ns, size)`
triples; `value` is whatever `analyze()` returned. Derived, disposable, and
regenerated on any mismatch.

## Resolution of the task doc's first open question

> Do checkpoints currently store any Python object that has no natural array
> encoding?

**No blocking case.** Every leaf is an ndarray, a Python/NumPy scalar, a string,
`None`, or a container (`dict`, `list`, `tuple`) of those. Four shapes need an
explicit schema rule rather than a natural HDF5 mapping:

1. **`None`-valued array slots** (`E_grid_brem`, `brem_wide`, and ~8 case
   fields). HDF5 has no null dataset. Encoded as an explicit null marker, not an
   empty array — an empty array is a distinct, reachable value.
2. **Absent-vs-null** (`spec_coherent`, and the 12 conditional case keys from
   `payload-inventory.md`). Absence is identity-significant; the encoding must
   not normalise it to a stored null.
3. **`tuple` vs `list`** — the case payload holds `hkl_list` as a list of
   3-tuples and `long_offsets_fs` as a tuple. `case_content_key` hashes through
   sorted JSON so tuple/list does not affect identity, but `to_dict()` legacy
   byte-fixtures do. The schema tags sequence kind.
4. **Non-string mapping keys** — the store is keyed by `E0_keV` (`float`), and
   records are keyed by material name (`str`). HDF5 group names are strings, so
   float keys need a lossless round-trip (repr-based, plus a stored key-kind
   tag).

No callable, no live handle, no class instance with custom `__reduce__` appears
in any persisted payload. The one class instance that *is* pickled anywhere —
`campaign.sweep.Sweep` in `package_smoke.py` — is a test fixture, not stored
state.

## Consequence for naming

`.pkl` filenames stay. 136 literal `.pkl` references across 22 modules encode
the CAS and component layout, which this task holds fixed; `discover()` globs
`*.pkl` and `checkpoint_exists` stats `line.pkl`. `_checkpoint_io.load` already
sniffs magic bytes rather than trusting the extension, across three
generations (plain pickle, gzip, zstd). HDF5 becomes the fourth generation read
by the same sniff. The extension is a path token, not a format claim — a point
the schema document has to state plainly, because `h5py.File("line.pkl")` is
what an outside reader will be told to type.
