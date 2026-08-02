# CLI command-home consolidation

## Scope

Slice 1 of [`docs/plans/cli-redesign-implementation-plan.md`](../../../docs/plans/cli-redesign-implementation-plan.md): put every `cxr` subcommand's Click wiring under `src/cxr_mc/cli/commands/`, while keeping reusable/domain logic in its existing owner. This is a pure internal refactor: command spelling, help, exit codes, output, public domain imports, and the `_entry.scan` shim remain unchanged.

## Decision

Use `cli/commands/`, not a flat `cli/`: the existing `cli/` package has 16 modules and P1 adds 13 command modules, exceeding the RFC's flat-package threshold before the move lands.

## Checklist

- [ ] Add `cli/commands/` package and move standalone command wiring.
- [ ] Split `scan.py` and `blaze.py` so drivers remain domain-owned and Click wiring moves.
- [ ] Update lazy dispatch without changing surface behavior.
- [ ] Preserve or deliberately replace every test import seam.
- [ ] Regenerate `docs/cli-reference.md` and `docs/repo_map.md`.
- [ ] Pass CLI reference/export-freeze, targeted CLI, lint, typecheck, and relevant full tests.

## Non-goals

- No noun-to-verb surface change, output-format redesign, compatibility removal, energy-grid rename, recompute/prune fold, or artifact-store work.
- No physics/domain package moves.

## Acceptance evidence

- `docs/cli-reference.md` is unchanged except generated-path references required by the internal move, and its freeze test passes.
- Root help remains lazy; all command paths and deprecated aliases preserve current behavior.
- Existing export-freeze guards, targeted CLI tests, lint/typecheck, and the relevant suite pass.
