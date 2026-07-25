---
name: todo-sync
description: Use when work addresses/changes TODO items or reconciling TODO.md between main and active cxr-mc branches, especially across linked worktrees.
---

# TODO Sync

Treat `main:TODO.md` + its header convention as authoritative. Preserve others'
working trees; edit only `TODO.md`.

## Workflow

1. Work addressing/changing tracked item: inspect active-branch `TODO.md` +
   `main:TODO.md` before completion. Remove completed detail; update/remove main
   summary based on remaining work.
2. Require clean starting tree for cross-branch edits; never stash/discard
   unrelated work.
3. Read `main:TODO.md`; remove merged items. Keep each surviving main item to
   one summary line with branch or design-document pointer.
4. Preserve full text for items marked `>user<` while no task branch exists.
   During reconciliation, fold text into branch TODO, remove marker, then
   replace main entry with one-line summary once branch exists.
5. Run `git worktree list --porcelain` before touching any task branch.
6. If branch already has linked worktree, edit there. Otherwise switch only
   from clean main checkout.
7. Keep each task branch's `TODO.md` branch-scoped; preserve richer current
   details and repair stale paths/names.
8. Commit doc-only changes separately per branch. Do not push. Return the main
   checkout to `main` and report commit ids plus ahead-of-origin counts.

Large read-only inventory: cheap fresh subagent may inspect branches/worktrees.
Keep one explicit `TODO.md` writer; no parallel edits.

## Stop conditions

- A working tree is dirty with changes outside this task.
- Branch ownership or the authoritative task text is ambiguous.
- A required worktree is inaccessible.

Skip unchanged branches; never manufacture empty commits.
