---
name: todo-sync
description: Use when checking that main's TODO.md is accurate and that the TODO.md merge driver is installed; branch copies are disposable and auto-resolve to main on merge/rebase.
---

# TODO Sync

`main:TODO.md` is the single source of truth. Branch `TODO.md` copies are
disposable: the `.gitattributes` `TODO.md merge=ours` driver resolves every
conflicting hunk to the current branch (main's copy when a task branch merges
in, or the rebase base) with no manual resolution. You no longer force
byte-equality across branches.

## When invoked

1. Confirm the merge driver is installed in this clone:
   `git config --local --get merge.ours.driver` must print `true`. If missing,
   run `uv run cxr-dev bootstrap` (see `tasks/README.md`). Without it, git falls
   back to a normal 3-way merge and TODO.md conflicts return.
2. Read `main:TODO.md` once. Verify it is accurate: one summary line per active
   item with a branch and `tasks/<branch-name>/` pointer; `>user<` text
   preserved exactly until triaged.
3. Drop or rewrite lines only on `main`, never propagated from a branch — the
   driver discards conflicting branch edits, so a completed-task removal that
   overlaps a main edit is only reliable when authored on `main`. Use `triage`
   for `>user<` extraction; `dispatch-task` retire owns removing a landed task's
   line.

Do not copy `TODO.md` between branches, stash, discard work, or edit task/docs
content. Touch only `main:TODO.md`. Non-conflicting branch edits (e.g. a branch
removing only its own pointer line) apply cleanly on merge and need no action.
