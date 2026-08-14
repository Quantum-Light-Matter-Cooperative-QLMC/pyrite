# Result persistence format

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
- Interchange was evaluated, then deferred after the stored payload proved to
  be a spectral tally rather than a photon phase-space record.

Out of scope, deliberately:

- **The CAS layout, sharding, atomic-write discipline, and lock model.**
  Retained unchanged. Only the leaf encoding changes.
- Bulk rewriting of existing archives. Opportunistic only.
- Any interchange import/export, `PhotonSource` implementation, or downstream
  transport integration.

## Why interchange is deferred

PyRITE persists energy-binned spectral tallies, not individual photons. A
particle interchange record needs physical semantics for position, direction,
energy, time, polarization, and statistical weight. Exporting the current
payload would therefore invent a source model rather than translate data.

Future work begins with a concrete downstream consumer, defines a format-neutral
`PhotonSource` contract and closure tests, then selects an adapter. MCPL remains
a strong candidate for Geant4, OpenMC, MCNP/PHITS, and McStas/McXtrace, but it
is not selected or required by this task. Interchange can support geometry
downstream of a defined source surface; it cannot add arbitrary target geometry
to PyRITE's transport and emission physics.

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
- [x] B — Add `h5py` as a required dependency; write the
      result-format ADR recording the HDF5 decision and its reasoning.
- [x] C — Write the versioned schema document under `docs/repo-design/storage/`,
      including dataset names, dtypes, units, and the `identity_version` /
      `schema_version` fields.
- [x] D — Writer for the new format; reader dispatching on format.
- [x] E — Reader shim for `.pkl`, permanent. Opportunistic rewrite on next save.
- [x] F — Dropped/deferred after re-evaluation: define a format-neutral
      `PhotonSource` from a concrete consumer before choosing or validating an
      interchange adapter. No synthetic MCPL particle list is produced here.
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
- **Re-evaluated (2026-08-14): no interchange dependency.** Format support is
  not the blocker; the stored tally lacks the phase-space semantics a particle
  format requires. `mcpl` was removed. MCPL remains a candidate after a
  consumer-driven `PhotonSource` contract exists.
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
- **Resolved by deferral:** PyRITE has no emitted-photon record. The runner
  persists only energy-binned spectra plus aggregate metadata; emission
  positions, individual directions, times, and polarization are discarded
  before checkpoint assembly. An MCPL projection would therefore require a new
  sampling/estimator contract and would invent or choose phase-space fields.
  Slice F is dropped rather than silently defining a lossy projection. Future
  work must identify the consumer, define `PhotonSource` semantics and closure
  tests, and only then select an adapter.

## Implementation evidence (2026-08-13)

- `252fd46` — originally added required h5py/MCPL dependencies and ADR-0009;
  the 2026-08-14 re-evaluation retains h5py and removes MCPL.
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
- Interchange intentionally deferred; no synthetic/fabricated particle records
  were written.

## Re-evaluation evidence (2026-08-14)

- Removed the direct `mcpl` dependency; regenerated `uv.lock` removed `mcpl`,
  `mcpl-core`, and `mcpl-python` while retaining `h5py`.
- Amended ADR-0008 without erasing its original rationale; revised ADR-0009 and
  Change 6 of the RFC to make the consumer-driven `PhotonSource` ordering the
  current decision.
- `uv lock --check`: passed. Packaging suite: 243 passed. Offline docs: passed.
  Full `pyrite-dev verify`: passed, including the complete test suite, docs,
  lint, typecheck, import contracts, repository-map check, and skill sync.

## Delegation slices and required skills

- A → `implement-task`; `repo-orientation`. Inventory; no behavior change.
- B → `implement-task-lite`; `documentation-maintenance`. `one-shot`: the format
  decision is made, so this is dependency lines plus an ADR transcribing the
  recorded reasoning.
- C → `lead-task`; `documentation-maintenance`. The schema is the durable
  artifact and needs slice A's inventory in hand.
- D, E → `implement-task`; `catalog-golden` + `regression`.
- F → dropped/deferred. A later consumer-driven task must define
  `PhotonSource` before selecting an adapter or validation tool.
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
- Round-trip through the new format reproduces every stored spectrum bit for
  bit.
- The versioned schema is documented under `docs/repo-design/storage/` and the
  result-format ADR exists.

## Related

- `refactor/typed-case-record` — supplies `identity_version`, which the new
  schema records. RFC sequencing prerequisite.
- ADR-0008 retains arbitrary target geometry as a non-goal while its dated
  amendment defers interchange format selection until a real downstream
  consumer and photon-source boundary exist.
- TODO Active item 1, physics validation ledger — archival robustness of the
  evidence files is the practical motivation.
