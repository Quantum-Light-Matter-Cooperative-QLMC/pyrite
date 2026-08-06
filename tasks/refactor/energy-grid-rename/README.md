# energy-grid module rename (CLI-redesign slice 2)

Worktree: `/home/alexa/dev/cxr-mc-energy-grid-rename` — branch `refactor/energy-grid-rename` (off `main` @ 2de2a33).

## Scope

Slice 2 of [`docs/plans/cli-redesign-implementation-plan.md`](../../../docs/plans/cli-redesign-implementation-plan.md):
collapse `line_grid/` + `_energy_grid.py` into one `energy_grid/` package to match
the `cxr energy-grid` surface noun. Ungated, parallel with slice 1. Pure internal
rename — no command surface, help, output, exit, or checkpoint-format change.

## Design decision (beyond a bare `git mv`)

The old `line_grid/__init__.py` was the 707-line **Click command group** (eager
`import click`, `cxr_mc.remote`, `cxr_mc.cli`). The Monte Carlo hot path
(`runner`/`sweep`/`run`) imports the lightweight `_energy_grid` encoding helpers.
Folding encoding into the package while the heavy group stays in `__init__` would
make every scan-worker startup execute the Click/remote import chain — a
regression. So:

- Command group moved `__init__.py` → `energy_grid/_command.py` (verbatim; internal
  `cxr_mc.line_grid` imports rewritten to `cxr_mc.energy_grid`).
- New thin `energy_grid/__init__.py` exposes `command` **lazily** via `__getattr__`
  (same seam slice 1 used for scan/blaze) so hot-path `energy_grid.encoding`
  imports stay Click-free.
- `_energy_grid.py` → `energy_grid/encoding.py`.

## Done

- [x] `git mv line_grid → energy_grid`; `__init__.py → _command.py`; `_energy_grid.py → encoding.py`.
- [x] Thin `energy_grid/__init__.py` with lazy `command` `__getattr__`.
- [x] Rewrote import token `cxr_mc.line_grid` → `cxr_mc.energy_grid` and the three
  `_energy_grid import` forms → `energy_grid.encoding` across all `src`/`tests` `.py`
  (only the path token — bare local vars/fixtures named `line_grid` untouched, e.g.
  `runner.py:1149`, `test_sweep.py` param). Verified: zero residual references.
- [x] Renamed `tests/test_line_grid_*.py` → `tests/test_energy_grid_*.py` (8 files).
- [x] Curated docs: `docs/repo_map.md` (cli/energy_grid entry), `docs/api.md`
  (`cxr_mc.energy_grid`).

### Intentionally NOT touched
- `data/materials.toml` comment (`line_grid_bounds_job.py` — old script name, not a path).
- `tests/data/cli_contract.json` help-text example (`combined_line_grid_bounds.json`
  is an output filename, not a module path) — keeps the CLI freeze test green.
- `docs/package-structure-rfc.md` — historical proposal; point-in-time record of the pre-move layout.

## Verification (run 2026-08-04, all green)

- [x] Lint: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache UV_PROJECT_ENVIRONMENT=/tmp/cxr-mc-venv uv run cxr-dev lint`
- [x] Typecheck: `... cxr-dev typecheck` (no stale `cxr_mc.line_grid` type refs).
- [x] Targeted tests (CPU): `... CXR_MC_BACKEND=cpu uv run cxr-dev test tests/test_energy_grid_apply.py tests/test_energy_grid_bounds.py tests/test_energy_grid_cli.py tests/test_energy_grid_defaults.py tests/test_energy_grid_derive.py tests/test_energy_grid_golden.py tests/test_energy_grid_job.py tests/test_energy_grid_provenance.py tests/test_sweep.py tests/performance/test_profile.py tests/test_cli_completion.py tests/test_cli_json.py tests/test_cli_json_wiring.py tests/test_cli_profile.py tests/materials/test_material_catalog.py tests/remote/test_remote.py tests/test_scan_quick_energy.py tests/test_altair_plots.py tests/test_altair_detectors.py` — **825 passed**.
- [x] Suites: packaging **183**, cli **890**, core **967 passed / 40 skipped**, apps **287**.
- [x] CLI reference/export freeze green (zero surface change), inside the cli suite.
- [x] Checkpoint commit on `refactor/energy-grid-rename`.
- [ ] On land: mark slice 2 done in the implementation-plan sequence table (like slice 0).

### Fixes the verification pass turned up

The mechanical token rewrite missed three spots; all fixed in this branch:

1. **`from cxr_mc import line_grid` form** (module-object import, not a dotted path)
   in `tests/test_energy_grid_cli.py` and `tests/test_cli_json_wiring.py`. The CLI
   test also monkeypatched `line_grid.remote` / `.emit_json_result` / `.cli_json` /
   `._pull_combined` — those attributes now live on `energy_grid._command`, and the
   thin lazy `__init__` raises `AttributeError` for them, so the tests import
   `_command` directly and patch that seam (8 failures, incl.
   `test_click_follow_logs_propagates_remote_exit_status[1|75|130]`).
2. **`_dev.py:84`** cli-suite glob still listed `test_line_grid_cli.py` → renamed.
3. Stale path comments: `energy_grid/golden.py:11`, `remote.py:73`,
   `_remote/lifecycle.py:1161`.

## Non-goals
- No command/help/output/exit change; `docs/cli-reference.md` unaffected.
- No move of CLI wiring into `cli/commands/` (that's slice 1/3 territory); the group
  stays domain-adjacent in `energy_grid/_command.py`.
