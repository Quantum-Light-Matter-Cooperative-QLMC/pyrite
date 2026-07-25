---
name: todo-sync
description: Use when reconciling TODO.md between main and active cxr-mc task branches, especially when branches may already be checked out in linked worktrees.
---

# TODO Sync

Treat `main:TODO.md` and its header convention as authoritative. Preserve other
people's working trees and edit only `TODO.md`.

## Workflow

1. Require a clean starting tree; never stash or discard unrelated work.
2. Read `main:TODO.md`, remove merged items, and keep each surviving main item to
   one summary line with its branch or design-document pointer.
3. For each item marked `>user<`, preserve its full text while no task branch
   exists; fold it into the branch TODO during steps 1-2, then remove the marker
   and replace the main entry with a one-line summary once the branch exists.
4. Run `git worktree list --porcelain` before touching any task branch.
5. If a branch is already in a linked worktree, edit it there. Otherwise switch
   only from a clean main checkout.
6. Keep each task branch's `TODO.md` scoped to that branch, preserving the richer
   current details and repairing stale paths or names.
7. Commit doc-only changes separately per branch. Do not push. Return the main
   checkout to `main` and report commit ids plus ahead-of-origin counts.

## Stop conditions

- A working tree is dirty with changes outside this task.
- Branch ownership or the authoritative task text is ambiguous.
- A required worktree is inaccessible.

Skip unchanged branches; never manufacture empty commits.
