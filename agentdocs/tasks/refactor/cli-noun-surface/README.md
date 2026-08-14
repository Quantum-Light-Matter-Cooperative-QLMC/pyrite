# Reduce the CLI noun surface

Branch: `refactor/cli-noun-surface`
Source: [core architecture RFC](../../../../docs/repo-design/core-architecture-rfc.md),
Change 7 — sequencing step 7.
Depends on: `refactor/scene-object-model` (so every removed command has a
documented public-API replacement). Partially separable.

## Problem and scope

Thirteen top-level groups and 88 documented commands, targeting nine. Current
groups: `run`,
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

Nine top-level nouns, down from thirteen: `run`, `app`, `checkpoint`, `config`,
`remote`, `job`, `profile`, `material`, `beam`.

| Current | Disposition |
| --- | --- |
| `run`, `job`, `app`, `profile`, `material`, `beam` | Retained |
| `checkpoint` | Retained under its own name |
| `performance` | Moved to `pyrite-dev` |
| `energy-grid derive\|show\|defaults` | Retained under `material` / `profile` as grid inputs |
| `energy-grid verify\|gc\|regen-golden\|add\|rm` | Moved to `pyrite-dev` |
| `config`, `setup`, `completion` | Consolidated under `pyrite config` |
| `remote` | Retained as resource management only; `--remote` stays the run modifier |

The reduction comes from machinery leaving the user CLI entirely, not from
merging physical nouns together.

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

`feature/named-beam-objects` (COMPLETE 2026-08-09) deliberately promoted `beam`
to a first-class CLI noun. **Resolved on review: `beam` stays.** The RFC's first
draft proposed folding it into `profile`; that was rejected. See the decisions
section below.

## Implementation path

`src/pyrite/cli/commands/` (4 926 lines across 20 modules) plus the lazy-dispatch
group in `src/pyrite/cli/`. All removals go through the existing D7 deprecation
harness with a hidden warning alias for one support window.

## Checklist

- [x] A — Map every one of the 88 commands to keep / move / deprecate, with the
      replacement spelling for each move. No command is retired without a named
      replacement, and "the replacement is `pyrite-dev`" counts only if the
      command is genuinely maintenance rather than a user workflow.
      The current post-beam tree is 94 visible paths; the complete disposition
      and invariant inventory is in [command-disposition.md](command-disposition.md).
- [ ] B — Move `performance` to `pyrite-dev`, with hidden deprecated aliases for
      the old spellings.
- [ ] C — Relocate `energy-grid derive|show|defaults` under `material` /
      `profile`; move `verify` / `gc` / `regen-golden` / `add` / `rm` to
      `pyrite-dev`. Coordinate with `refactor/detector-scorer`, which demotes
      the grid to a detector input.
- [ ] D — Consolidate `setup` and `completion` under `pyrite config`.
- [ ] E — Decide and execute on the interactive profile-mutation flows.
      Separate slice; it is the largest line-count item and the least settled.
- [ ] F — Regenerate `docs/repo-design/cli/cli-reference.md` and
      `cli-deprecations.md`; the freeze test guards each step.
- [ ] G — Write the ADR amending ADR-0002.

## Decisions and open questions

- **Decided:** every existing CLI contract is retained unchanged.
- **Decided:** no retired spelling breaks without a warning window under D7.
- **Decided (review): `beam` stays a top-level noun.** It is a catalog object
  with named entries, profile reference counts, and its own lifecycle —
  structurally identical to `profile` and `material`. Folding it would also
  contradict `refactor/scene-object-model`, which promotes `pr.Beam` to one of
  three public API primitives; removing the CLI noun for one of the three
  top-level physical objects while elevating it in Python is incoherent. After
  `refactor/detector-scorer`, a `detector` noun is a plausible tenth.
- **Decided (review): no `cache` noun is created.** The first draft would have
  merged `checkpoint`, `performance`, and `energy-grid` maintenance under
  `cache`, but those are not one kind of artifact. Checkpoints are *results* —
  GPU-hours to produce, not cheaply regenerable, and the evidence behind
  validation ledger rows; naming that surface `cache` invites someone to
  discard it. Energy-grid and performance artifacts genuinely are derived, so
  they leave the user CLI for `pyrite-dev`. This answers TODO UI backlog item 3
  (`gc` confusion) rather than renaming around it.
- **Open:** whether interactive profile mutation is removed, reduced, or kept.
  Note TODO Inbox item 1 requests `--lock`/`--unlock` for profile mutability,
  and Bugs item 2 reports `profile create --from` not copying materials — both
  imply continued investment in that surface. Reconcile before slice E.
- **Open:** how much of this is safely landable *before*
  `refactor/scene-object-model`. Slices B, C, D, and F are largely mechanical
  and may not need to wait; slice E does, because "use the Python API instead"
  is only an honest answer once the API exists.

### Open-question resolution (2026-08-14)

- **Interactive profile mutation stays.** Removing leaf verbs would not reduce
  the top-level noun count, named-beam attachment depends on the lifecycle, and
  the approved lock/unlock and create-from follow-ups explicitly invest in it.
  Those untriaged/bug follow-ups remain out of scope here.
- **Slices B-D and F-G are safely landable.** This branch now contains the
  completed detector-scorer and scene-object-model stacks, including the public
  filesystem-free simulation replacement. No pre-Scene exception remains.
- The exact command/alias and contract inventory is checkpointed before behavior
  work in [command-disposition.md](command-disposition.md).

## Delegation slices and required skills

- A → `lead-task`; `cli-ui-ux`. The disposition map gates B–E.
- B, C, D → `implement-task`; `cli-ui-ux`. Regenerate the reference in the same
  change as each move.
- E → `lead-task`; `cli-ui-ux`. Material decisions open; never `one-shot`.
- F → `implement-task-lite`; `cli-ui-ux` + `documentation-maintenance`.
- G → `implement-task-lite`; `documentation-maintenance`.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev cli-reference --write
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev cli-deprecations --write
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite cli
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev docs
```

- Root help lists at most nine primary nouns.
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
