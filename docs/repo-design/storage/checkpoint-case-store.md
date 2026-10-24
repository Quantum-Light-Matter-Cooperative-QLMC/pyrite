# Cross-profile checkpoint case store

Named profiles retain separate checkpoint identities while sharing numerically identical computed cases through a per-material content-addressable store (CAS). This avoids recomputing overlap such as the common energy range in `sub_100keV` and `sub_200keV`.

## Identity and layout

`profiles.case_content_key()` hashes schema `cxr.case-content-key.v1` plus the fully resolved case dictionary. Profile labels, fidelity/variant labels, execution chunk sizes, analytic mosaic post-processing, and reporting-only flux/current fields are excluded. Physics inputs, Monte Carlo mosaic width, and RNG seed remain keyed.

Each raw transport output is stored once at:

```text
<checkpoint-root>/<material>/<first-two-hex>/<sha256>.pkl
```

Writes use same-filesystem temporary files plus `os.replace`. A content key names deterministic output, so independent writers need no shared lock. `run_sweep()` records a thin `<stem>/cases.json` (or `<stem>.cases.json` for a legacy file path) mapping each requested case to its content key.

The `.pkl` suffix is a stable layout token. New component, shard, and CAS leaf files contain [versioned HDF5](result-schema.md); the shared reader continues to accept all historical pickle encodings.

## Cache modes

| Mode | Read CAS | Write CAS | Per-profile checkpoint |
|---|---:|---:|---:|
| default | yes | yes | yes |
| `--recompute` | no | yes | yes |
| `--no-cache` | no | no | yes |
| `pyrite-dev perf` without explicit cache flag | no | no | yes |

`--no-cache` and `--recompute` are mutually exclusive. Explicit `--recompute` overrides the performance default, including Nsight re-exec.

## Compatibility and migration

Component checkpoints are materialized under `checkpoints/<stem>/{line,brem,characteristic}.h5`. `line.h5` is the authoritative case index and contains PXR/CBS line spectra; `characteristic.h5` is an optional companion attached on read as its own `spec_characteristic` array, `channeling.h5` likewise carries `spec_channeling` and is written only when a record has it, and `brem.h5` carries the continuum. No merged record array includes another component. Analysis, archive, prune, slim, and remote pull retain their merged in-memory contracts; manifest-only consumers belong to the checkpoint-command rework.

Legacy plain, gzip, and zstd pickle artifacts are never rewritten merely by reading them. Any later normal save uses HDF5, so migration is opportunistic and inherits the existing component/CAS atomic replacement discipline.

A compatible existing profile checkpoint seeds missing CAS blobs during normal resume. Migration compares requested and stored case content keys before writing, so legacy records without sufficient identity metadata recompute instead of poisoning the shared store.

Profile labels stay in dataset identity and provenance but not per-case content identity. Two profiles reuse a case only when resolved inputs, including seed, hash identically.

## Selective reads

Internal checkpoint consumers can call `_checkpoint_store.load_selected(stem,
root, cases, components=...)`, where `cases` is a set of `(config_name,
energy_keV)` pairs. The line component always supplies the authoritative
record index; `components` selects which companions attach. Completed-config
shards override component files per case, with the same component-grid checks
as full loading. Missing cases produce no entries; a missing checkpoint raises
`FileNotFoundError` for a nonempty request.

Bare schema-v2 HDF5 reads traverse only requested record payloads, including
columnar tables. Key columns and layout metadata still scale with the file's
record count. Legacy containers and compressed transfer frames use the existing
full-decode compatibility path before filtering. The default in-memory sweep API remains unchanged.

The synthetic storage benchmark `checks/checkpoint_selective_memory.py` compares
one-case and full-store reads using identical warm file caches and fresh reader
processes, reporting wall time and peak RSS. It measures storage reads, not
campaign execution or consolidation.

## Low-retention sweeps

`pyrite.runs.run.run_sweep(..., retain_results=False)` opts into disk-backed
result retention. The default remains `True`. The low-retention mode requires
an empty `results` mapping and a component checkpoint path; explicit legacy
`.pkl` output paths are rejected.

After successful shard and manifest writes, each completed record in `results`
contains only `case` and `checkpoint_path`. Callers obtain spectra through
checkpoint loading. Group callbacks still receive config names, but must load
their spectra explicitly instead of reading arrays from `results`. Progress,
budgets, observation production, dataset identity checks, and CAS reuse retain
their normal semantics.

Each completed energy is written immediately, preserving other already stored
energies for that config. A failed write leaves the live record's arrays in
`results`. This also makes partially completed configs recoverable after an
interruption. Rewriting the config shard for each energy trades additional I/O
for bounded retention; the transient payload bound is one config, not one case.
Case dictionaries, record keys, manifest audits, and HDF5 indexing metadata
remain resident or scale with the campaign size.

Resume reads case metadata without spectra from bare v2 HDF5. Missing CAS blobs
are seeded from selective case reads; cache hits are persisted and released
one case at a time. Legacy input containers retain the full-decode fallback
described above.

Final consolidation loads one config at a time and writes all component
staging files from that payload. Each staged component uses the existing v2
columnar record encoding and shared array pool. All staging files finish before
any authoritative component is replaced; companions publish before the line
index. Shards are removed only after the final manifest succeeds. A staging,
publication, or manifest failure therefore leaves recovery shards available.

`checks/checkpoint_sweep_memory.py` measures retained and disk-backed sweeps at
16, 64, and 256 configs, two energies per config, and 16,384 spectral bins. It
builds real HOPG campaign cases but substitutes deterministic synthetic spectra
to isolate host retention and storage overhead from transport. It reports peak
RSS at the case frontier and after consolidation, wall time, checkpoint phase
timing, process read/write bytes, and final artifact size. Each arm has a fresh
process and checkpoint directory, matched scalar warmup, and an 8 GiB virtual
memory cap. These are orchestration measurements, not transport predictions.

A three-repeat Linux/Python 3.14.4 run of that workload measured:

| Records | Peak RSS, retained / disk (MiB) | Wall time, retained / disk (s) |
|---|---|---|
| 32 | 476.06–476.20 / 475.77–476.32 | 0.33–0.35 / 1.34–1.44 |
| 128 | 495.77–496.29 / 495.39–495.88 | 1.35–1.40 / 5.80–6.23 |
| 512 | 838.09–838.32 / 548.25–548.46 | 7.06–7.49 / 28.36–29.41 |

At 512 records, case-frontier peak RSS was 703.09–703.43 MiB retained versus
536.91–537.36 MiB disk-backed. Process write volume was 863.21 versus
1,312.01–1,312.02 MiB, read volume 0.62 versus 956.89 MiB, and final artifacts
194.96 versus 268.36 MiB. Per-config column tables add metadata relative to one
campaign-wide table; early persistence rewrites config shards. Interpreter
startup and HDF5 indexing overhead dominate smaller workloads. Peak RSS includes
those costs; the mode bounds spectral retention, not total metadata growth.

## Version control

Nothing under `checkpoints/` is tracked except the `.gitkeep` files that hold `checkpoints/` and `checkpoints/archive/` open. Payloads (`*.pkl`), CAS blobs, `*meta.json`, `cases.json`, and `cxr.lock.json` are all local run output, reproducible from a profile plus seed, and are ignored.

Five `cxr.lock.json` files were tracked between 2026-05 and 2026-08-09 without an intended rule; each was swept into an unrelated commit. They were untracked on 2026-08-09. A tracked lock is worse than no lock: it announces a completed dataset whose payload is ignored, so a fresh clone sees a GC root for blobs it does not have.

## Ownership and regression surface

- `src/pyrite/campaign/profiles.py`: content-key schema and canonical hashing.
- `src/pyrite/checkpoints/_checkpoint_store.py`: sharded paths and atomic blob I/O.
- `src/pyrite/checkpoints/_checkpoint_io.py`: versioned HDF5 codec and permanent legacy-pickle reader dispatch.
- `src/pyrite/runs/run.py`: replay/write, migration, and `cases.json` manifests.
- `src/pyrite/runs/scan.py` and `src/pyrite/cli/commands/scan.py`: run policy plus CLI cache modes and performance options.
- `tests/materials/test_profiles.py`, `tests/scan/test_run.py`, `tests/cli/test_local_click_cli.py`, and `tests/scan/test_scan_beam_options.py`: identity, reuse, migration, flag, and forwarding regressions.
