# CLI command-home consolidation

## Scope

Slice 1 of [`docs/plans/cli-redesign-implementation-plan.md`](../../../docs/plans/cli-redesign-implementation-plan.md): put every `cxr` subcommand's Click wiring under `src/cxr_mc/cli/commands/`, while keeping reusable/domain logic in its existing owner. This is a pure internal refactor: command spelling, help, exit codes, output, public domain imports, and the `_entry.scan` shim remain unchanged.

## Decision

Use `cli/commands/`, not a flat `cli/`: the existing `cli/` package has 16 modules and P1 adds 13 command modules, exceeding the RFC's flat-package threshold before the move lands.

## Checklist

- [x] Add `cli/commands/` package and move standalone command wiring.
- [x] Split `scan.py` and `blaze.py` so drivers remain domain-owned and Click wiring moves.
  Wiring → `cli/commands/{scan,blaze}.py`; drivers keep `run`/`_run_json`/`main`.
  Domain modules re-export `command` via a lazy module `__getattr__` (no import
  cycle) so `scan.command`/`blaze.command` seams and dispatch keep resolving.
- [x] Update lazy dispatch without changing surface behavior. `cli/__init__.py`
  `run` and `cli/commands/material.py` `blaze` now point at `cli.commands.*`.
- [x] Preserve or deliberately replace every test import seam. Preserved via the
  `__getattr__` seams — command-test files use `scan.<driver>`/`scan.command`
  from the same name, so no test edits needed.
- [x] `docs/cli-reference.md` current (generator `--check` passes; zero surface
  change).
- [x] `docs/repo_map.md` updated. It is curated prose, but it named the moved
  modules directly (`### cli/energy_grid.py, cli/profile.py, cli/material.py`,
  `### cli/checkpoint.py`, `### cli/performance.py`) — those paths are now
  7-line alias shims, so the map would have sent readers to the wrong files.
  Added a `### cli/commands/` section documenting the one-module-per-group
  layout and the `sys.modules` aliases, repointed the entry-point lines at
  `cli.commands.*`, and refreshed the `scan.py`/`blaze.py` entries (their
  `Public:` lists advertised an `add_subparser` that does not exist on `main`
  either — pre-existing staleness, corrected while here).
- [x] Pass CLI reference/export-freeze, targeted CLI (890 passed), lint, typecheck.
  Also fixed 2 pre-existing ty errors (`profile.py`/`sweep.py` importing through
  the `sys.modules`-swap alias modules) by repointing at `cli.commands.*`.

## Independent verification (2026-08-04)

Re-run in a clean env (`UV_PROJECT_ENVIRONMENT=/tmp/cxr-mc-venv-cch`), all green:
lint, typecheck, suites packaging **183**, cli **890**, core **967 passed / 40
skipped**, apps **287**. Packaging re-run after the `repo_map.md` edit: 183.
No code defects found; the only gap was the doc above.

## Non-goals

- No noun-to-verb surface change, output-format redesign, compatibility removal, energy-grid rename, recompute/prune fold, or artifact-store work.
- No physics/domain package moves.

## Acceptance evidence

- `docs/cli-reference.md` is unchanged except generated-path references required by the internal move, and its freeze test passes.
- Root help remains lazy; all command paths and deprecated aliases preserve current behavior.
- Existing export-freeze guards, targeted CLI tests, lint/typecheck, and the relevant suite pass.

## Delegation

- `wip/cli-command-home-standalone`: move standalone Click-only commands without
  changing root lazy dispatch; checkpoint commit only.
- `wip/cli-command-home-scan-blaze`: split the `run` and blazed-sweep Click
  wiring from their headless drivers; checkpoint commit only.
- `wip/cli-command-home-existing`: relocate the existing flat `cli/` command
  modules into `cli/commands/` without changing root lazy dispatch; checkpoint
  commit only.
