# Cross-profile checkpoint case store

Named profiles retain separate checkpoint identities while sharing numerically
identical computed cases through a per-material content-addressable store
(CAS). This avoids recomputing overlap such as the common energy range in
`sub_100keV` and `sub_200keV`.

## Identity and layout

`profiles.case_content_key()` hashes schema `cxr.case-content-key.v1` plus the
fully resolved case dictionary. Profile labels, fidelity/variant labels,
execution chunk sizes, analytic mosaic post-processing, and reporting-only
flux/current fields are excluded. Physics inputs, Monte Carlo mosaic width,
and RNG seed remain keyed.

Each raw transport output is stored once at:

```text
<checkpoint-root>/<material>/<first-two-hex>/<sha256>.pkl
```

Writes use same-filesystem temporary files plus `os.replace`. A content key
names deterministic output, so independent writers need no shared lock.
`run_sweep()` records a thin `<stem>/cases.json` (or `<stem>.cases.json` for a
legacy file path) mapping each requested case to its content key.

The `.pkl` suffix is a stable layout token. New component, shard, and CAS leaf
files contain [versioned HDF5](result-schema.md); the shared reader continues to
accept all historical pickle encodings.

## Cache modes

| Mode | Read CAS | Write CAS | Per-profile checkpoint |
|---|---:|---:|---:|
| default | yes | yes | yes |
| `--recompute` | no | yes | yes |
| `--no-cache` | no | no | yes |
| `-p/--perf` without explicit cache flag | no | no | yes |

`--no-cache` and `--recompute` are mutually exclusive. Explicit
`--recompute` overrides the performance default, including Nsight re-exec.

## Compatibility and migration

Component checkpoints remain materialized under
`checkpoints/<stem>/{line,brem}.pkl`. Analysis, archive, prune, slim, and remote
pull retain their existing read contracts; manifest-only consumers belong to
the checkpoint-command rework.

Legacy plain, gzip, and zstd pickle artifacts are never rewritten merely by
reading them. Any later normal save uses HDF5, so migration is opportunistic and
inherits the existing component/CAS atomic replacement discipline.

A compatible existing profile checkpoint seeds missing CAS blobs during normal
resume. Migration compares requested and stored case content keys before
writing, so legacy records without sufficient identity metadata recompute
instead of poisoning the shared store.

Profile labels stay in dataset identity and provenance but not per-case content
identity. Two profiles reuse a case only when resolved inputs, including seed,
hash identically.

## Version control

Nothing under `checkpoints/` is tracked except the `.gitkeep` files that hold
`checkpoints/` and `checkpoints/archive/` open. Payloads (`*.pkl`), CAS blobs,
`*meta.json`, `cases.json`, and `cxr.lock.json` are all local run output,
reproducible from a profile plus seed, and are ignored.

Five `cxr.lock.json` files were tracked between 2026-05 and 2026-08-09 without
an intended rule; each was swept into an unrelated commit. They were untracked
on 2026-08-09. A tracked lock is worse than no lock: it announces a completed
dataset whose payload is ignored, so a fresh clone sees a GC root for blobs it
does not have.

## Ownership and regression surface

- `src/pyrite/campaign/profiles.py`: content-key schema and canonical hashing.
- `src/pyrite/checkpoints/_checkpoint_store.py`: sharded paths and atomic blob I/O.
- `src/pyrite/checkpoints/_checkpoint_io.py`: versioned HDF5 codec and permanent
  legacy-pickle reader dispatch.
- `src/pyrite/runs/run.py`: replay/write, migration, and `cases.json` manifests.
- `src/pyrite/runs/scan.py` and `src/pyrite/cli/commands/scan.py`: run policy
  plus CLI cache modes and performance options.
- `tests/materials/test_profiles.py`, `tests/scan/test_run.py`,
  `tests/cli/test_local_click_cli.py`, and `tests/scan/test_scan_beam_options.py`:
  identity, reuse, migration, flag, and forwarding regressions.
