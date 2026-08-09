---
name: todo-sync
description: Use when main's TODO.md accuracy or the local TODO merge driver needs a read-only audit; never edit backlog or task files.
---

# TODO Sync

`main:TODO.md` is the single source of truth. Branch `TODO.md` copies are
disposable: the `.gitattributes` `TODO.md merge=ours` driver resolves every
conflicting hunk to the current branch (main's copy when a task branch merges
in, or the rebase base) with no manual resolution. You no longer force
byte-equality across branches.

1. Confirm the merge driver is installed in this clone:
   `git config --local --get merge.ours.driver` must print `true`. If missing,
   run `uv run cxr-dev bootstrap` (see `agentdocs/README.md`). Without it, git falls
   back to a normal 3-way merge and TODO.md conflicts return.
2. Read `main:TODO.md` once. Verify it is accurate: one summary line per active
   item with a branch and `agentdocs/tasks/<branch-name>/` pointer; `>user<` text
   preserved exactly until triaged.
3. Report stale/missing pointers and merge-driver failures without editing.
   `triage` owns new pointers; `dispatch-task` retirement owns completed-item
   removal.

Do not compare branch copies for equality, copy TODO between branches, stash or
discard work, or edit any file.
