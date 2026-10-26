# Trajectory artifact storage

`pyrite.montecarlo.trajectories` writes one HDF5 file per captured case. The user workflow is in [Saved trajectory workflows](../../reference/trajectory-workflows.md#save-electron-trajectories); the authoritative layout is the module docstring. This page records why schema 3 stores row fields the way it does and what that was measured to cost.

## Schema 3 layout decisions

Schema 3 changes only how row fields (one entry per segment) sit on disk. `read_trajectory_artifact(...).transport` still returns the transported mapping with the same dtype, shape, row order and values. Spectra replayed from a schema-3 artifact equal those from schema 2.

- **Chunked, compressed rows.** By default each row dataset uses 4,096-row chunks with the `shuffle` filter and LZF (`_ROW_CHUNK`, `_ROW_FILTERS`). Rows are written in bounded blocks aligned to whole chunks, so device-resident captures still download one block at a time.
- **Per-electron index.** `/electron_index` holds `electron_id`, `row_start` and `row_stop` for each stored history run. Streaming (`iter_electron_row_blocks`), history selection (`select_trajectories` with histories, `--first` or `--sample`) and export read run boundaries from it instead of scanning or bisecting `electron_id`.
- **Per-history constants.** `t0_ang` is stored once per index entry (`pyrite_kind = "electron_rows"`). Before choosing that form, the writer checks that the value is bitwise identical across every row of each history. If any history varies, the field stays per-row.
- **Narrow IDs.** `electron_id`, `track_id`, `parent_id` and `flight_id` are stored as `int32`, and `substep_id` as `int16` (or `int32` when needed). Each narrowing applies only when the field's full range fits; the transported dtype is kept in `pyrite_dtype`. `/transport_order` and the index arrays use `int32` when they fit.
- **One reader seam.** Raw `/transport` datasets no longer equal the transported arrays, so every reader goes through `transport_view(handle)`. It restores the transported dtype, expands per-history fields to rows, and slices rows like an h5py dataset. Schema 1/2 nodes pass through it unchanged. The view caches node lookups, because h5py attribute and group access cost more than a small selected read.

## Capture storage policies

`TrajectoryCapture(root=..., storage_policy="compact")` is the default.
`storage_policy="throughput"` disables HDF5 compression and shuffle for
chunked transport and index datasets while retaining schema-3 electron-major
layout, integer narrowing, per-history compaction, aliases, and atomic
publication. The direct `write_trajectory_artifact(..., storage_policy=...)`
entry point accepts the same setting. Existing readers need no policy
argument; the selected policy is stored in the HDF5 root attribute
`storage_policy`. Storage policy does not affect transport or scoring.

Use `throughput` when GPU transport outpaces LZF compression and disk space is
less important than capture latency. The schema-2-like uncompressed reference
below is **not** identical to the new schema-3 throughput policy: the latter
retains compact indexing and ID storage. Its actual write/read times, peak
host RAM and VRAM, and transfer volume must be measured on the target host
before changing the default. No new measurements are claimed here.

## Benchmarks

### Case (a): CPU sweep case

Case (a) is a 2,000-electron, 30 keV silicon sweep case: 2 µm slab, 30° tilt, midpoint energy model, default coupled radiative fields, lockstep CPU transport. It produced 294,506 segments, and its transport took 1.72 s. Times are the best of three on the development laptop with a warm page cache. "Select 10" selects and reads 10 histories spread across the file.

Filter and chunk choice at 4,096-row chunks, with the schema-2 writer for reference:

| Layout | Bytes/segment | Write (share of transport) | Full read | Stream (65,536-row blocks) | Select 10 |
|---|---|---|---|---|---|
| schema 2 (uncompressed) | 207.7 | 0.070 s (4%) | 0.063 s | 0.044 s | 24.0 ms |
| **schema 3, LZF + shuffle** | **89.2** | **0.31 s (18%)** | 0.19 s | 0.16 s | 27.0 ms |
| schema 3, gzip-1 + shuffle | 84.6 | 0.50 s (29%) | 0.16 s | 0.13 s | 24.6 ms |
| schema 3, gzip-4 + shuffle | 83.4 | 0.58 s (34%) | 0.17 s | 0.14 s | 26.3 ms |

Effect of chunk size with LZF + shuffle:

| Chunk rows | 2,048 | 4,096 | 8,192 | 16,384 | 65,536 |
|---|---|---|---|---|---|
| Bytes/segment | 89.9 | 89.2 | 88.6 | 88.4 | 88.6 |
| Select 10 | 19 ms | 28 ms | 45 ms | 78 ms | 138 ms |

Decisions:

- **LZF over gzip.** Schema 3 cuts case (a) by 57% (207.7 to 89.2 B/segment). Only LZF keeps the write cost under a quarter of transport time. gzip saves another 5 to 6% of the file but costs 29 to 34% of transport.
- **Portability cost.** LZF ships with h5py but not with the HDF5 C library. Tools outside h5py, such as `h5dump`, HDFView or MATLAB, need the LZF filter plugin (for example the one in `hdf5plugin`) to read row datasets.
- **4,096-row chunks.** A selected history decompresses whole chunks, so selection time grows with chunk size, while compression barely improves past 4,096 rows. At 4,096 rows, a 10-history selection costs about the same as schema 2.

Where the bytes go: in case (a), the eight float64 row fields `r_mid`, `v_hat`, `L_ang`, `t_start_ang`, `t_end_ang`, `E_start_keV`, `E_end_keV` and `E_repr_keV` make up 82 of the 89 bytes per segment. Floating-point mantissas compress poorly without loss. Further savings would need a lossy representation, which schema 3 does not adopt. Sparse fields compress almost to nothing (hard-radiative vectors take 0.3 B/segment), as do narrowed IDs and the per-history `t0_ang`.

Full reads and streaming are about three times slower than schema 2 because of decompression. At under 1 µs per segment they remain small next to transport, which took about 6 µs per segment here.

### Case (b): large GPU capture

Case (b) is the `trajectory_demo` hopg geometry at 30 keV: 45° tilt, 180° azimuth, a 1 µm spot, 30,000 electrons, and a 20 µm (2×10⁵ Å) slab. It ran on the remote GPU box (RTX 5080, 8 CPU cores, shared and time-sliced) with the CUDA transport core and produced 6,869,145 segments. Transport took 2.0 s. Each layout was timed once, so the timings are noisy. "Uncompressed" is the schema-3 writer with filters, ID narrowing and per-history storage turned off, which is the schema-2 row layout.

| Layout | Size | Bytes/segment | Write (share of transport) | Full read | Stream (1 Mi / 64 Ki rows) | Select 10 |
|---|---|---|---|---|---|---|
| uncompressed (schema-2-like) | 1.28 GiB | 199.7 | 1.55 s (76%) | 0.69 s | 0.67 / 1.18 s | 10 ms |
| schema 3, host segments | 0.57 GiB | 89.4 | 7.0 s (345%) | 5.6 s | 5.1 / 3.7 s | 38 ms |
| schema 3, device-resident segments | 0.57 GiB | 89.4 | 5.7 s (248%) | 3.1 s | 3.5 / 3.2 s | 162 ms |

The size reduction carries over at scale: 55%, from 1.28 to 0.57 GiB. The write budget does not. LZF compresses about 1 million segments per second on this host's CPU, but the CUDA core transports about 3.4 million per second. A schema-3 capture therefore costs several times the GPU transport it records, against about 0.8× uncompressed. Case (a)'s 25% bound holds only for CPU transport. Ways to bring GPU captures back under budget, none adopted here:

- the new opt-in `throughput` policy, which keeps schema-3 compaction while removing HDF5 compression (GPU timings still to be measured);
- compressing on a writer thread that overlaps the spectrum phase;
- storing floats at lower precision.

Host and device-resident captures both read back equal to the transported arrays.

## Compatibility

New runner captures record a data-only resolved physics view in
`settings.resolved_physics`: descriptor IDs, model-generation markers, original
case parameters, and per-case consumed-table digests. Full and bounded spectrum
replay compare this snapshot with current resolution before scoring. Changed
models or table digests require new trajectories. Backend diagnostics and
descriptor documentation are outside this comparison.

Captures without this optional snapshot retain their historical replay checks;
readers do not populate omitted selectors with current defaults or infer past
table inventories. Geometry-only reading and export remain available regardless
of current model or table availability. The snapshot does not change the case
digest or the HDF5 schema version.

Readers accept schemas 1 to 3. Schema-2 files keep their plain row datasets and have no `/electron_index`. Selection and streaming then fall back to bisection and boundary scans, and give the same selections and blocks. `tests/data/trajectory_artifacts/schema2_hopg_lockstep.h5` is a genuine schema-2 capture. The tests check that it replays bit for bit, both directly and after being rewritten as schema 3. `tests/helpers/trajectory_schema.py` rewrites any new artifact into the schema-1 or schema-2 layout for compatibility tests.
