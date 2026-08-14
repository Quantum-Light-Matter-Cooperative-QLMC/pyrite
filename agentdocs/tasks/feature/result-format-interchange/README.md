# Result persistence format and MCPL interchange

Branch: `feature/result-format-interchange`
Source: [core architecture RFC](../../../../docs/repo-design/core-architecture-rfc.md),
Change 6 — sequencing step 6.
Depends on: `refactor/typed-case-record` (step 1). Blocks: nothing.

## Problem and scope

Checkpoints are pickles — `checkpoints/<stem>/{line,brem}.pkl` — plus
content-addressed case blobs, campaign locks, and manifests. The store *design*
is sound; see
[checkpoint case store](../../../../docs/repo-design/storage/checkpoint-case-store.md)
and
[dataset identity and storage](../../../../docs/repo-design/storage/dataset-identity-and-storage.md).

Pickle is the problem: version-fragile across Python and NumPy releases, unsafe
to accept from a third party, and unreadable by any other tool. It is a poor
archival format for results that back scientific claims — and this repository
maintains a physics validation ledger whose evidence lives in those files.

In scope:

- Array payloads move to **HDF5** under a documented, versioned schema.
- A reader shim keeps existing `.pkl` datasets loadable **indefinitely**.
  Migration is opportunistic — rewrite on next save, matching what the store
  already does for the legacy flat layout.
- MCPL export for the emitted photon list, and optionally for the transport
  segment list.

Out of scope, deliberately:

- **The CAS layout, sharding, atomic-write discipline, and lock model.**
  Retained unchanged. Only the leaf encoding changes.
- Bulk rewriting of existing archives. Opportunistic only.
- MCPL *import*. Export is the escape hatch the RFC needs; import is a separate
  question with its own scope.

## Why MCPL

[MCPL](https://mctools.github.io/mcpl/mcpl.pdf) is the interchange format shared
by Geant4, MCNP, McStas, and McXtrace. It is the pragmatic answer to "can
another code model our detector?" — it makes that possible without PyRITE owning
a general detector geometry, which the RFC records as an explicit non-goal. This
slice is what buys the right to say no to arbitrary geometry.

## Implementation path

Likely owners:

| Concern | Location |
| --- | --- |
| Checkpoint read/write | `src/pyrite/checkpoints/_checkpoint_io.py` (104 lines), `_checkpoint_store.py` (286 lines) |
| Archive / slim | `src/pyrite/checkpoints/archive.py`, `slim.py` |
| Lock metadata | `src/pyrite/checkpoints/campaign_lock.py` |
| Result assembly | `src/pyrite/results/store.py`, `selection.py` |
| Producers of pickled state | `src/pyrite/runs/run.py`, `src/pyrite/montecarlo/runner/__init__.py`, `src/pyrite/remote/lifecycle.py` |
| CLI touchpoints | `src/pyrite/cli/commands/{scan,blaze,slim}.py` |

`rg -l pickle src/` currently reports fifteen modules; slice A must classify each
as result payload, control-plane state, or incidental.

## Checklist

- [x] A — Classify every current pickle use. Only *result array payloads* are in
      scope; control-plane state (job records, remote lifecycle) is a separate
      question and should be listed, not converted.
      Evidence: [`pickle-inventory.md`](pickle-inventory.md) — 15 modules
      classified; 4 payload, 2 control-plane, 9 incidental; all persistence
      funnels through `_checkpoint_io`.
- [x] B — Add `h5py` and `mcpl` as required dependencies; write the
      result-format ADR recording the HDF5 decision and its reasoning.
- [x] C — Write the versioned schema document under `docs/repo-design/storage/`,
      including dataset names, dtypes, units, and the `identity_version` /
      `schema_version` fields.
- [x] D — Writer for the new format; reader dispatching on format.
- [x] E — Reader shim for `.pkl`, permanent. Opportunistic rewrite on next save.
- [ ] F — MCPL export for the emitted photon list. Validate with `mcpltool`.
- [x] G — Dropped: transport segments are transient kernel inputs and are not
      persisted; exporting them is not near-free.
- [x] H — Update the two storage design pages and `docs/repo_map.md`.

## Decisions and open questions

- **Decided:** existing `.pkl` datasets load forever. No flag day, no bulk
  migration script as a prerequisite.
- **Decided:** CAS, locks, sharding, atomic writes unchanged.
- **Decided (review): HDF5, not Zarr.** Zarr's advantage is many concurrent
  writers into one large array, which does not arise here — PyRITE writes many
  small independent per-case artifacts. Against that, Zarr's
  directory-of-chunks layout fights the store's write-then-rename atomicity,
  where a single `.h5` renames atomically and maps one-to-one onto a CAS blob.
  The decision is atomicity and blob semantics, not reader popularity.
- **Decided (review): `mcpl` is a required dependency**, the full meta-package.
  `mcpl-core` ships prebuilt wheels for macOS x86-64/arm64, manylinux
  x86-64/aarch64, musllinux, and Windows amd64/arm64; `mcpl-python` is
  pure Python needing only `numpy>=1.22`; all Apache-2.0. There is no build risk
  for an extra to hedge against, and a conditionally available escape hatch
  would weaken the arbitrary-geometry non-goal it is supposed to justify.
  Do not split it into `mcpl-python` alone to save a wheel.
- **Resolved (slice A):** no checkpoint stores a Python object without a natural
  array encoding. Every leaf is an ndarray, a scalar, a string, `None`, or a
  container of those. Four shapes need an explicit schema *rule* rather than a
  natural mapping — `None`-valued array slots, absent-vs-null keys
  (`spec_coherent`), tuple-vs-list sequence kind, and non-string mapping keys
  (`E0_keV` is a float). See [`pickle-inventory.md`](pickle-inventory.md).
- **Decided (slice A): on-disk filenames stay `.pkl`.** 136 literal `.pkl`
  references across 22 modules encode the CAS and component layout this task
  holds fixed. `_checkpoint_io.load` already dispatches on magic bytes across
  three generations; HDF5 is the fourth read by the same sniff. The extension is
  a path token, not a format claim.
- **Open:** MCPL carries particle type, position, direction, energy, time,
  weight, and polarization. Confirm PyRITE's emitted-photon record maps onto
  that without loss, and document any field that does not survive the round
  trip.
- **Blocked after inspection:** PyRITE has no emitted-photon record. The runner
  persists only energy-binned spectra plus aggregate metadata; emission
  positions, individual directions, times, and polarization are discarded
  before checkpoint assembly. An MCPL projection would therefore require a new
  sampling/estimator contract and would invent or choose phase-space fields.
  Slice F cannot be implemented losslessly without a scope/design decision.
  The minimum upstream decision is either (1) specify an owning phase-space
  estimator that retains photon position, direction, energy, time,
  polarization, and statistically normalized weight, or (2) explicitly redefine
  F as a lossy weighted spectral-bin projection and specify every default field
  plus its normalization. This task does neither implicitly.

## Implementation evidence (2026-08-13)

- `252fd46` — required h5py/MCPL dependencies and ADR-0009.
- `4bb595f` — schema-version-1 HDF5 writer, signature dispatch, permanent
  plain/gzip/zstd pickle readers, h5py-independent-inspection regression, and
  bit-for-bit spectrum regression.
- `e782cce` — lower validation reproduction cache dataclass keys to native
  scalar tuples, restoring them at the cache boundary and retaining legacy
  dataclass-key pickle reads.
- Focused checkpoint/run/analysis regression surface: 221 passed.
- Full `pyrite-dev test`: passed. `pyrite-dev verify`: passed, including the
  repeated full suite, docs, lint, typecheck, import contracts, repo-map check,
  and skill synchronization. Standalone `pyrite-dev docs`: passed.
- MCPL validation: blocked; no scientifically defined photon phase-space record
  exists to export. No synthetic/fabricated particle records were written.

## Delegation slices and required skills

- A → `implement-task`; `repo-orientation`. Inventory; no behavior change.
- B → `implement-task-lite`; `documentation-maintenance`. `one-shot`: the format
  decision is made, so this is dependency lines plus an ADR transcribing the
  recorded reasoning.
- C → `lead-task`; `documentation-maintenance`. The schema is the durable
  artifact and needs slice A's inventory in hand.
- D, E → `implement-task`; `catalog-golden` + `regression`.
- F → `implement-task`; `scientific-library`. `one-shot` once C lands, since the
  acceptance check is external (`mcpltool`) and unambiguous.
- G → `implement-task-lite`; drop rather than stretch.
- H → `implement-task-lite`; `documentation-maintenance`.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev docs
```

- A result file is readable by `h5py` with **no PyRITE import**.
- Every stored `.pkl` dataset loads unchanged.
- An exported MCPL file passes `mcpltool` validation.
- Round-trip through the new format reproduces every stored spectrum bit for
  bit.
- The versioned schema is documented under `docs/repo-design/storage/` and the
  result-format ADR exists.

## Related

- `refactor/typed-case-record` — supplies `identity_version`, which the new
  schema records. RFC sequencing prerequisite.
- The RFC's arbitrary-geometry non-goal cites MCPL export as the escape hatch;
  `refactor/target-geometry-surface` slice F should link here once this lands.
- TODO Active item 1, physics validation ledger — archival robustness of the
  evidence files is the practical motivation.
