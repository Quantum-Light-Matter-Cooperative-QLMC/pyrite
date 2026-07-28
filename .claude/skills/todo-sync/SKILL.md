---
name: todo-sync
description: Use when work addresses/changes TODO items or reconciling TODO.md between main and active cxr-mc branches, especially across linked worktrees.
---

# TODO Sync

`TODO.md` is a **single shared backlog, identical on every branch and on
`main`** — one summary line per item + a pointer. Branch-scoped detail lives in
`tasks/<branch-leaf>.md`, never in `TODO.md` and never in `docs/`. This is
policy **A**: because branch `TODO.md` never diverges from `main`, a
fast-forward (branch→branch or `main`→branch) cannot silently clobber the
backlog. Treat `main:TODO.md` as authoritative; preserve others' working trees;
edit only `TODO.md` and `tasks/*`.

**Why it matters:** a fast-forward moves the ref with **no merge**, so any
per-branch divergence in a tracked file like `TODO.md` is overwritten on the
next ff. Keeping it identical everywhere is the fix — do not reintroduce
branch-specific `TODO.md` bodies.

## Workflow

1. Reconcile against `main:TODO.md` + the item's `tasks/<branch-leaf>.md`.
   `TODO.md` edits must keep branch == `main`; never write branch-only content
   into `TODO.md`.
2. Require a clean starting tree for cross-branch edits; never stash/discard
   unrelated work.
3. Read `main:TODO.md`; slim completed items. Keep each surviving item to one
   summary line with a `→ feature/<branch>; tasks/<branch-leaf>.md` pointer.
4. Preserve full text for items marked `>user<` while no task branch exists.
   During reconciliation, fold text into `tasks/<branch-leaf>.md`, remove the
   marker, then replace the `TODO.md` entry with a one-line summary once the
   branch exists.
5. Run `git worktree list --porcelain` before touching any task branch.
6. If a branch has a linked worktree, edit there. Otherwise switch only from a
   clean `main` checkout.
7. Keep branch-scoped detail in `tasks/<branch-leaf>.md`, not `TODO.md`; repair
   stale paths/names there.
8. **Landing a branch into `main` (branch to be dropped):** promote any durable
   design/physics from `tasks/<branch-leaf>.md` into a proper `docs/` note, then
   `git rm tasks/<branch-leaf>.md`, and slim the `TODO.md` item to completion.
9. Commit doc-only changes separately per branch. Do not push. Return the main
   checkout to `main` and report commit ids plus ahead-of-origin counts.

Large read-only inventory: cheap fresh subagent may inspect branches/worktrees.
Keep one explicit `TODO.md` writer; no parallel edits.

## Stop conditions

- A working tree is dirty with changes outside this task.
- Branch ownership or the authoritative task text is ambiguous.
- A required worktree is inaccessible.
- An edit would make branch `TODO.md` diverge from `main` (policy A violation) —
  put the detail in `tasks/<branch-leaf>.md` instead.

Skip unchanged branches; never manufacture empty commits.
