# Content-addressed CLI artifact model (redesign slice 6)

Paused implementation handoff: [`HANDOFF.md`](HANDOFF.md).

Branch: `feature/cli-artifact-model`; based on slice-5 closure commit `7b19402`
because slice 5 is not yet merged to `main` and slice 6 requires slices 3-5.

Implements slice 6 of
[`agentdocs/plans/cli-redesign-implementation-plan.md`](../../../plans/cli-redesign-implementation-plan.md)
and accepted ADR
[`0003`](../../../../docs/adr/0003-content-addressed-artifact-model.md).

## Authority and boundaries

- Implement and checkpoint coherent, verified phases on this branch.
- Do not push, edit `main:TODO.md`, merge, or remove either worktree.
- Preserve existing checkpoint case-CAS and `checkpoint gc` semantics.
- Pre-existing strict-Sphinx, CuPy JIT, geometry typing, and toctree failures are
  accepted baseline limits; new or changed-surface failures are not.
- No physics equations or Monte Carlo kernels change.

## Pinned first-cut decisions

1. Energy-grid artifacts use schema `cxr.energy-grid-artifact.v1`. The frozen
   hash identity is canonical JSON over schema, material key, sorted line-grid
   definitions, bremsstrahlung grid, and derived beam energies. Provenance,
   notes, timestamps, ref names, and orphan metadata are excluded annotations.
2. Immutable JSON objects live in a sharded store beside the catalog under
   `energy-grid-artifacts/<first2>/<sha256>.json`. Atomic create-if-absent plus
   byte/hash verification provides dedup and corruption detection.
3. `profiles.NAME.energy_grid_refs` is the sole new mutable grid pointer map.
   Resolution order is explicit profile ref, then legacy `[energy_grids.NAME]`
   or `[energy_grids.standard]` plus profile brem fields as a read-only D7
   compatibility fallback. Reads never auto-migrate.
4. Canonical `energy-grid add` creates/deduplicates artifacts and atomically
   repoints one resolved profile. Retired `energy-grid apply` warns and
   redirects to the same ref behavior; it never mutates scan ranges, standard
   overrides, or legacy grid payloads.
5. Canonical `energy-grid verify` checks referenced/stored object hashes.
   `energy-grid gc` reclaims unreachable objects after 14 days. Orphan
   timestamps live outside artifact bytes; `--prune-all` bypasses age but keeps
   preview/TTY/`-y` safety and revalidation.
6. A completed dataset writes deterministic `cxr.lock.json` beside its
   checkpoint metadata. Active and archived lockfiles are artifact GC roots.
7. Remote sync compares artifact hashes and transfers missing immutable objects;
   existing checkpoint checksum and case-CAS behavior remains unchanged.

## Acceptance

- [x] Equal identity inputs deduplicate; any identity input change changes hash.
- [x] Hash-input and excluded-annotation sets are frozen by tests.
- [x] Catalog resolves profile refs and legacy fallback equivalently.
- [x] `energy-grid add` repoints refs without changing legacy/standard scan data.
- [x] `apply` and `line delete` remain warning D7 shims with removal metadata.
- [x] Per-run lockfile records resolved profile, artifact hashes, and dataset
      identity; incomplete runs do not claim completion.
- [x] `verify` detects missing, corrupt, and mismatched artifacts.
- [x] `gc` honors reachability, 14-day grace, `--prune-all`, exact preview, and
      fail-closed revalidation.
- [x] Remote artifact sync skips hashes already present and transfers missing
      hashes only.
- [x] Generated CLI reference, CLI contract, deprecation table, repo map, lint,
      focused tests, and CLI suite pass except confirmed allowed baseline failures.

## Implementation checkpoints

- [x] Store, canonical hash contract, verification, unit tests.
- [x] Catalog ref schema and legacy dual-read migration tests.
- [x] Canonical add/apply ref-repointing and profile lifecycle integration.
- [x] Run lockfile emission and GC reachability/grace.
- [x] Remote hash-diff sync and migration aliases.
- [x] Generated artifacts and verification matrix.

## Verification evidence (2026-08-05, resumed session)

Local environment has no CuPy, so `tests/montecarlo/*` and everything importing
the runtime fail to even collect. To separate that baseline from real
regressions, every suite was run twice under the import-only stub
(`PYTHONPATH=/tmp/cxr-cupy-import-stub`): once on this branch and once on a
throwaway clone at the base commit `7b19402`. The failure sets are identical.

| Suite | Base | Branch |
|---|---|---|
| core | 66 failed | 66 failed (same test ids) |
| cli | 1 failed (`test_mott_missing_table_logs_debug_once`) | same |
| apps | 1 failed (`test_matplotlib_vacuum_legs_...`) | same |
| packaging | 2 failed (`test_agent_session_start_...`, `test_all_matches_frozen_set`) | same |

Those stub-run failures are stub artifacts or pre-existing; the stub is import
scaffolding only and is never evidence about GPU behavior. Also green:
`cxr-dev lint`, `cxr-dev format`, `git diff --check`, and `cxr-dev typecheck`
(9 remaining diagnostics, all baseline CuPy imports plus the geometry typing
waiver — every diagnostic in this branch's new or changed code was fixed).
