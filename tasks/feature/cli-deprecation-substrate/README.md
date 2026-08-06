# CLI deprecation substrate (RFC D7)

## Scope

Prerequisite for [`docs/plans/cli-redesign-implementation-plan.md`](../../../docs/plans/cli-redesign-implementation-plan.md)
slice 3 (surface reshuffle): every renamed/retired `cxr` spelling keeps
working for a published support window, emits one stderr warning naming its
replacement, and appears in a generated reference doc, per RFC D7
([`docs/cli-redesign-rfc.md`](../../../docs/cli-redesign-rfc.md) L224-232).

## Design

`src/cxr_mc/cli/_deprecations.py` owns the harness: a frozen `Deprecation`
registry keyed by command path (`SUPPORT_WINDOW_MINORS = 2`, deprecated in
`0.1.0` -> removed in `0.3.0`), `DeprecatingGroup(click.Group)` warning once
per invocation in `resolve_command` (before callbacks/`--help`, via a
`ctx.meta` guard), and `invocation_path()` walking parent contexts to build
the warned path. `SELF_WARNING` covers the two `sweep show`/`sweep set`
leaves whose replacement depends on the arguments given, so those callbacks
warn themselves and `DeprecatingGroup` skips them.

Wired into `cli/_core.py` (`LazyGroup(DeprecatingGroup)`), `cli/commands/profile.py`,
`cli/commands/sweep.py`, `energy_grid/_command.py`, `_remote/cli.py`.
`scripts/generate_cli_deprecations.py` regenerates
[`docs/cli-deprecations.md`](../../../docs/cli-deprecations.md) from the
registry (`--write`/`--check`, mirrors `generate_cli_reference.py`).

## Checklist

- [x] `_deprecations.py` harness + registry (25 rows) + wiring across all
  deprecated command groups.
- [x] `tests/cli/test_deprecations.py`: registry/live-tree agreement in both
  directions, hidden-command coverage, single-warning-per-path, `--help`
  emits none, support-window arithmetic, generated doc currency.
- [x] `docs/cli-deprecations.md` generator + checked-in output current.
- [x] `docs/cli-reference.md` prose header points at `cli-deprecations.md`;
  regenerated.
- [x] `docs/repo_map.md`: ownership entry for `cli/_deprecations.py`.
- [x] Fixed pre-existing CPU-import breakage found while regenerating docs:
  `montecarlo/spectrum.py` had an unguarded, uncalled `@xp.fuse() def
  fused_prefactor` (dead code duplicating `_brem_dsigma_dk_core`'s inline
  prefactor) that raised `AttributeError` on any NumPy-only backend since
  NumPy has no `.fuse`. Deleted; unrelated to D7 but blocked every CLI test
  and the reference-doc generator on this machine.
- [x] Full verify: `cxr-dev lint` clean, `typecheck` clean, `test-suite cli` 942 passed, `test-suite packaging` 183 passed, `test-suite apps` 273 passed. `test-suite core` 984 passed / 39 skipped / 4 failed — the 4 failures (`test_adaptive_chunk_*` in `tests/montecarlo/test_montecarlo.py`) are pre-existing, unrelated to D7 or the `fused_prefactor` deletion: they hardcode a fp32 (`_REAL_BYTES=4`) chunk-size constant that only holds on a GPU backend, and this machine has none installed, so `REAL` resolves to fp64. Only visible now because the collection-blocking bug above previously aborted the whole file before these ran. Left unfixed as out of scope for this task.

## Non-goals

No noun-to-verb surface reshuffle (plan slice 3), no artifact-store or
energy-grid work (slices 5-6). This slice only builds the compatibility
mechanism slice 3 will lean on when old spellings retire.
