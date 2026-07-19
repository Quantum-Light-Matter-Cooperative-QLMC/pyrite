# Module-boundaries refactor handoff

Date: 2026-07-19

## Location

- Branch: `refactor/module-boundaries`
- Worktree: `/home/alexa/dev/cxr-mc/.worktrees/refactor-module-boundaries`
- State: uncommitted WIP; main checkout untouched

## Intended scope

First low-risk slices from repository refactor review:

1. Extract material-catalog grid/descriptor decoding behind the existing
   `cxr_mc.materials.catalog` API.
2. Extract pure remote status/presentation helpers behind the existing
   `cxr_mc.remote` facade.

Do not expand this handoff into Monte Carlo, sweep, notebook, test-file, or
shell-generation refactors until these slices are verified and committed.

## Completed: catalog decoder

- Added `src/cxr_mc/materials/_catalog_decode.py`.
- `materials/catalog.py` imports decoder helpers while retaining public record
  classes, `MaterialConfigError`, `_negative`, CIF loading, parser orchestration,
  lazy default-catalog loading, and public `LineGridByEnergy` availability.
- Preserves `CrystalSpec.__module__ == "cxr_mc.materials.catalog"`, package
  identity, lazy manifest loading, exact error aggregation, and immutable grids.
- Verified by subagent: 45 tests passed across
  `test_material_catalog.py`, `test_catalog_startup_errors.py`, and
  `test_materials_package.py`; Ruff and `git diff --check` passed.

## WIP: remote presentation extraction

- Added `src/cxr_mc/_remote/__init__.py` and
  `src/cxr_mc/_remote/presentation.py`.
- `remote.py` currently aliases extracted constants/helpers from
  `_remote.presentation`.
- Builder was interrupted before verification. Treat this slice as unverified.
- Audit facade monkeypatch compatibility before keeping direct aliases. Example:
  tests may patch `remote._color_enabled`, while `_presentation._paint` resolves
  `_presentation._color_enabled` instead. Similar internal helper-to-helper calls
  may bypass facade patches.
- Keep `_attach_progress_dashboard` and anything using facade-patched
  `_poll_chain`, `tqdm`, or `time.sleep` in `remote.py`.
- Do not change generated Bash, SSH, reservation, scheduler, clear/stop, or pull
  behavior in this slice.

## Required next steps

1. Review `git diff` and map every extracted presentation helper's global
   references. Restore wrapper/dependency indirection where facade monkeypatches
   must remain effective.
2. Run:

   ```bash
   rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_remote.py
   rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_material_catalog.py tests/test_catalog_startup_errors.py tests/test_materials_package.py tests/test_check_config.py
   rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py lint
   rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py typecheck
   ```

3. Regenerate `docs/repo_map.md` because new implementation modules were added:

   ```bash
   rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py repo-map
   ```

4. Run full `verify`, inspect final diff, then commit only this isolated branch.

## Environment note

Canonical `uv` setup in the fresh worktree initially hit restricted-DNS
dependency downloads. Network escalation began populating the shared cache but
was interrupted. Catalog checks succeeded through the existing main checkout
virtual environment. Re-run canonical commands; do not assume fresh-worktree
environment setup completed.

## Safety contracts

- Generated remote shell builders require executable Bash tests, not substring
  assertions alone.
- Remote destructive operations must remain fail-closed across scheduler errors,
  ambiguous submissions, and reservations.
- `jobs/reservations` is bookkeeping, never an implicit runnable job.
- Remote pull checksums compare the exact transferred artifact; deterministic
  gzip bytes remain part of correctness.
