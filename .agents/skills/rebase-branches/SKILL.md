---
name: rebase-branches
description: Use when bringing cxr-mc task branches up to date with main, rebasing each branch that rebases cleanly or with only trivial TODO.md conflicts and skipping or aborting branches with significant conflicts.
---

# Rebase Branches

Rebase eligible branches onto `main`. Cheap subagents do per-branch work; this
agent coordinates, judges, and reports. Never push: rebases rewrite history and
later require `--force-with-lease`, which needs explicit user authority.

## Inventory

1. `git fetch --all --prune`; inspect `git branch -vv`,
   `git worktree list --porcelain`, per-branch dirty status, and ahead/behind
   counts vs `main`.
2. Skip and report: branches not behind `main`, dirty worktrees, mid-rebase or
   mid-merge state, unclear ownership.
3. Remaining behind-`main` branches are candidates.

## Rebase per branch

Fan out one cheap subagent (Haiku/Luna tier) per candidate branch. Never two
agents in one worktree. Give each explicit authority: rebase one named branch,
resolve only `TODO.md` conflicts, abort on anything else, no push, no force,
no test runs.

Each subagent:

1. Preflight the conflict surface without mutating state, e.g.
   `git merge-tree --write-tree main <branch>`, and list conflicted files.
2. No conflicts, or conflicts confined to `TODO.md` → rebase
   (`git -C <worktree> rebase main`, or `git rebase main <branch>` for an
   un-checked-out branch). Resolve `TODO.md` by taking `main`'s version.
3. Any other conflicted file → `git rebase --abort`; do not attempt
   resolution. Significant means any conflict outside `TODO.md`.
4. A branch whose changes are fully contained in `main` is not a conflict;
   report it as a landed/retire candidate instead of rebasing.
5. Report: rebased (old→new head) | aborted (conflicted files) | retire
   candidate | skipped (reason).

## After

1. Invoke `todo-sync` to restore `TODO.md` equality across rebased branches.
2. Do not run test suites per branch; dispatch and workers own validation.
3. Report a table: branch → rebased old→new | skipped (reason) | aborted
   (conflict files) | retire candidate. List branches now diverged from their
   upstream (each needs `--force-with-lease`) and stop; push only on explicit
   user instruction.
