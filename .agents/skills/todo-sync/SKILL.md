---
name: todo-sync
description: Use when checking or restoring exact TODO.md consistency across cxr-mc main, task branches, and linked worktrees while preserving untriaged user-authored text.
---

# TODO Sync

Enforce one invariant: every branch `TODO.md` equals authoritative
`main:TODO.md`.

## Workflow

1. Inspect `git worktree list --porcelain` and each target status.
2. Read `TODO.md` from main worktree. Stop if it has unknown uncommitted edits;
   never stash, discard, or overwrite user work.
3. Compare target copies byte-for-byte. Preserve `>user<` text exactly.
4. Copy only authoritative `TODO.md` into divergent clean or same-writer
   worktrees.
5. Verify equality; report changed and blocked worktrees.

Touch only `TODO.md`. Do not edit task/docs content, create/drop branches or
worktrees, commit, push, rebase, or dispatch. Use `triage` for `>user<`
extraction; use `dispatch-task` for task lifecycle.
