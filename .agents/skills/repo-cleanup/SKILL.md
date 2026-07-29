---
name: repo-cleanup
description: Use when tidying cxr-mc git state — rebasing task branches onto main where conflicts are trivial, pruning stale worktrees and tracking refs, deleting fully merged local/remote branches, and similar safe git hygiene chores.
---

# Repo Cleanup

General-purpose git hygiene: bring branches current with `main` and clear
merged/stale refs and worktrees. Cheap subagents (Haiku/Luna tier) do
per-branch and per-item work; this agent coordinates, judges, and reports.
Never push except the merged-remote-branch deletion gated below; rebases
rewrite history and later require `--force-with-lease`, which needs explicit
user authority.

## Inventory

1. `git fetch --all --prune`; inspect `git branch -vv`,
   `git worktree list --porcelain`, per-branch dirty status, and ahead/behind
   counts vs `main`.
2. Skip and report: branches not behind `main`, dirty worktrees, mid-rebase or
   mid-merge state, unclear ownership.
3. Remaining behind-`main` branches are rebase candidates.

## Rebase per branch

Fan out one cheap subagent per candidate branch. Never two agents in one
worktree. Give each explicit authority: rebase one named branch, resolve only
`TODO.md` conflicts, abort on anything else, no push, no force, no test runs.

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

## Cleanup chores

Run after rebases, so retire candidates are included. Every deletion is gated
by an objective check; delegate per item to cheap subagents with one named
target each. Never `git branch -D`, never wildcard or bulk deletes, never
touch `main`. Record each deleted ref's tip SHA for recovery before deleting.

Order matters: worktrees before their branches, local before remote.

1. Stale worktree admin entries: `git worktree prune` (safe, no data loss).
2. Empty worktrees: remove only when `git status` is clean and its branch is
   fully merged into `main` (`git branch --merged main`) or gone. Use plain
   `git worktree remove`; a refusal means stop and report, never `--force`.
3. Fully merged local branches: from `git branch --merged main`, delete with
   `git branch -d` only. Skip `main`, skip any branch still checked out in a
   remaining worktree, skip anything with an open task doc not yet retired.
4. Remote prune: `git remote prune origin` drops stale tracking refs (safe).
5. Fully merged remote branches: from `git branch -r --merged main`, delete
   with `git push origin --delete <branch>`. Never `main`/`master`, never a
   branch whose local counterpart survived step 3. This is the only permitted
   push; the merged check gates it.

## After

1. Invoke `todo-sync` to restore `TODO.md` equality across rebased branches.
2. Do not run test suites per branch; dispatch and workers own validation.
3. Report a table: branch → rebased old→new | skipped (reason) | aborted
   (conflict files) | retire candidate; plus chores done per item (worktree
   removed, branch deleted, remote pruned/deleted) with recovery SHAs. List
   branches now diverged from their upstream (each needs
   `--force-with-lease`) and stop; push only on explicit user instruction.
