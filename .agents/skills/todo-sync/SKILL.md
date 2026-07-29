---
name: todo-sync
description: Use when checking or restoring exact TODO.md consistency across cxr-mc main, task branches, and linked worktrees while preserving untriaged user-authored text.
---

# TODO Sync

Enforce one invariant: every branch `TODO.md` equals authoritative
`main:TODO.md`.

## Workflow

1. Inspect `git worktree list --porcelain` and each target status.
2. Read every worktree's current `TODO.md`, including uncommitted content.
   Treat dirty state as reconciliation input, not a blocker.
3. Merge unique intended edits into the main worktree copy. Preserve `>user<`
   text exactly. Stop only when the same backlog item has incompatible edits
   that cannot be combined without choosing intent.
4. Copy the reconciled main `TODO.md` into every divergent worktree.
5. Verify equality; report changed and blocked worktrees.

Never stash or discard work. Touch only `TODO.md`. Do not edit task/docs
content, create/drop branches or worktrees, commit, push, rebase, or dispatch.
Use `triage` for `>user<` extraction; use `dispatch-task` for task lifecycle.
