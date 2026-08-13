# Reduce the CLI noun surface

Branch: `refactor/cli-noun-surface`
Source: [core architecture RFC](../../../../docs/repo-design/core-architecture-rfc.md),
Change 7 — sequencing step 7.
Depends on: `refactor/scene-object-model` (so every removed command has a
documented public-API replacement). Partially separable.

## Problem and scope

Thirteen top-level groups and 88 documented commands. Current groups: `run`,
`setup`, `app`, `checkpoint`, `completion`, `config`, `performance`, `remote`,
`job`, `energy-grid`, `profile`, `material`, `beam`.

A substantial fraction is store maintenance surfaced as user vocabulary —
`checkpoint gc|rm|merge|slim|recompute|archive|restore`,
`energy-grid add|rm|verify|gc|regen-golden`, `performance rm`,
`remote gc|prune-jobs`. A further part is a TOML editor implemented as a command
group: `src/pyrite/cli/commands/profile.py` is 1 169 lines over
`src/pyrite/campaign/profile_edit.py` at 465 lines.

**The CLI's contracts are not in question and must not change.** Exit-code
discipline, the versioned JSON envelope, `-o/--output` semantics,
destructive-operation previews, the deprecation registry, and the generated
reference are all retained. This task concerns the *number of nouns*, not their
quality.

## Target state

Six top-level nouns: `run`, `profile`, `material`, `job`, `app`, `cache`.

| Current | Disposition |
| --- | --- |
| `run`, `job`, `app`, `profile`, `material` | Retained |
| `checkpoint`, `performance`, `energy-grid` maintenance verbs | Consolidated under `cache` |
| `energy-grid derive|show|defaults` | Retained under `material` / `profile` as grid inputs |
| `config`, `setup`, `completion` | Retained; candidates for `pyrite config` consolidation |
| `remote` | Retained as resource management only; `--remote` stays the run modifier |
| `beam` | Folded into `profile` **unless** named beams prove independently useful |

Narrower proposal, to be decided rather than assumed: keep `profile show`,
`profile list`, `material show`, `material validate`; reconsider the interactive
`create` / `rename` / `delete` flows, which are the bulk of the 1 634 lines
across `profile.py` and `profile_edit.py`. The catalog is already
self-validating, so edit-then-validate is a defensible alternative to
prompt-driven mutation.

## Prior art in this repository — read before deciding

This is not the first CLI surface pass. The following are landed and their
reasoning is binding context, not a blank slate:

- [ADR-0002](../../../../docs/adr/0002-cli-surface-redesign.md) — the noun→verb
  redesign. Change 7 **amends** it; the RFC names this as a follow-up ADR.
- `agentdocs/plans/cli-redesign-implementation-plan.md` — the linearized
  sequence for the earlier redesign, with slices 0–6 landed.
- Retired task records: `feature/cli-surface-simplification`,
  `feature/cli-verb-collapse-module-fold`, `feature/cli-surface-reshuffle`,
  `feature/cli-command-home`, `feature/cli-vocab-controls`,
  `feature/cli-deprecation-substrate`.

**`feature/named-beam-objects` is marked COMPLETE (2026-08-09) and deliberately
promoted `beam` to a first-class CLI noun.** The RFC's "fold `beam` into
`profile`" line is in direct tension with recently landed, user-requested work.
Do not fold it on the strength of the RFC alone; this needs an explicit user
decision. It is the single largest open question in this task.

## Implementation path

`src/pyrite/cli/commands/` (4 926 lines across 20 modules) plus the lazy-dispatch
group in `src/pyrite/cli/`. All removals go through the existing D7 deprecation
harness with a hidden warning alias for one support window.

## Checklist

- [ ] A — Confirm the `beam` disposition with the user. Blocking.
- [ ] B — Map every one of the 88 commands to keep / move / deprecate, with the
      replacement spelling for each move. No command is retired without a named
      replacement.
- [ ] C — Land `cache` and move the maintenance verbs under it, with hidden
      deprecated aliases for the old spellings.
- [ ] D — Relocate `energy-grid derive|show|defaults` under `material` /
      `profile`; move `verify` / `gc` / `regen-golden` to `cache` or
      `pyrite-dev`. Coordinate with `refactor/detector-scorer`, which demotes
      the grid to a detector input.
- [ ] E — Decide and execute on the interactive profile-mutation flows.
      Separate slice; it is the largest line-count item and the least settled.
- [ ] F — `config` / `setup` / `completion` consolidation, if slice B still
      justifies it after C–E.
- [ ] G — Regenerate `docs/repo-design/cli/cli-reference.md` and
      `cli-deprecations.md`; the freeze test guards each step.
- [ ] H — Write the ADR amending ADR-0002.

## Decisions and open questions

- **Decided:** every existing CLI contract is retained unchanged.
- **Decided:** no retired spelling breaks without a warning window under D7.
- **Open, blocking:** `beam`. See the prior-art note above.
- **Open:** does `cache` read as the right noun for `checkpoint` operations?
  Checkpoints are results, not a cache, and calling them cache invites users to
  treat them as discardable. `store` or `artifact` may be truer. Decide in
  slice B — it is a user-facing vocabulary choice, and TODO UI backlog item 3
  already records confusion about what `gc` means and whether it crosses
  profiles.
- **Open:** whether interactive profile mutation is removed, reduced, or kept.
  Note TODO Inbox item 1 requests `--lock`/`--unlock` for profile mutability,
  and Bugs item 2 reports `profile create --from` not copying materials — both
  imply continued investment in that surface. Reconcile before slice E.
- **Open:** how much of this is safely landable *before*
  `refactor/scene-object-model`. Slices C, D, and G are largely mechanical and
  may not need to wait; slice E does, because "use the Python API instead" is
  only an honest answer once the API exists.

## Delegation slices and required skills

- A → `lead-task`; requires a user decision, not an agent one.
- B → `lead-task`; `cli-ui-ux`. The disposition map gates C–F.
- C, D → `implement-task`; `cli-ui-ux`. Regenerate the reference in the same
  change.
- E → `lead-task`; `cli-ui-ux`. Material decisions open; never `one-shot`.
- F → `implement-task`; `cli-ui-ux`.
- G → `implement-task-lite`; `cli-ui-ux` + `documentation-maintenance`.
- H → `implement-task-lite`; `documentation-maintenance`.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev cli-reference --write
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev cli-deprecations --write
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite cli
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev docs
```

- Root help lists at most six primary nouns.
- No retired spelling breaks without a warning window.
- The generated reference and its freeze test are updated in the same change as
  each move.
- Exit codes, JSON envelope, `-o/--output` semantics, and destructive previews
  are unchanged — verified by the existing CLI contract tests, not by
  inspection.
- ADR amending ADR-0002 exists and is listed in `docs/adr/index.md`.

## Related

- `refactor/scene-object-model` — supplies the replacement for anything removed.
- `refactor/detector-scorer` — demotes `energy-grid` from user vocabulary; this
  task performs the actual verb relocation.
- TODO Inbox item 1 (`--lock`/`--unlock` profiles), Bugs items 1–3, UI backlog
  item 3 (`gc` naming) all land on this surface. Fold them in rather than
  fighting them.
