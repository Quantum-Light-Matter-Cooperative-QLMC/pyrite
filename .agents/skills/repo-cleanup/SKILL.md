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
user authority. No subagent may check out or switch branches in an existing
checkout; anything needing a branch on disk gets a fresh temporary worktree
(see "Rebase per branch").

## Inventory

1. `git fetch --all --prune`; inspect `git branch -vv`,
   `git worktree list --porcelain`, per-branch dirty status, and ahead/behind
   counts vs `main`.
2. The controlling agent reads `main:TODO.md` and cross-references every
   branch against it: note which branches a TODO item points at (and that
   item's state), and flag any active branch not tracked in `TODO.md` —
   untracked branches are reported as triage candidates, never deleted.
3. Skip and report: branches not behind `main`, dirty worktrees, mid-rebase or
   mid-merge state, unclear ownership.
4. Remaining behind-`main` branches are rebase candidates.

## Rebase per branch

Fan out one cheap subagent per candidate branch. Never two agents in one
worktree. Give each explicit authority: rebase one named branch, resolve only
`TODO.md` conflicts, abort on anything else, no push, no force, no test runs.

Never check out or switch branches in an existing checkout — that disturbs
ongoing work. Any subagent that needs a branch on disk (e.g. to resolve a
`TODO.md` conflict mid-rebase) must first `git worktree add` a fresh
temporary worktree for that branch, work inside it, and `git worktree remove`
it when done (or on abort). Prefer ref-only operations (`git rebase main
<branch>`, `git merge-tree`) that need no checkout at all.

Each subagent:

1. Preflight the conflict surface without mutating state, e.g.
   `git merge-tree --write-tree main <branch>`, and list conflicted files.
2. No conflicts, or conflicts confined to `TODO.md` → rebase
   (`git -C <worktree> rebase main` in the branch's existing worktree, in the
   subagent's fresh worktree, or `git rebase main <branch>` for a branch
   checked out nowhere). The `TODO.md merge=ours` driver auto-resolves TODO.md
   to `main`'s version — no manual step. If the rebase still halts on TODO.md,
   the driver is not installed: run `uv run cxr-dev bootstrap`, then take
   `main`'s version and continue. (`git merge-tree` preflight may report TODO.md
   as conflicted even though the real rebase resolves it.)
3. Any other conflicted file → `git rebase --abort`; do not attempt
   resolution. Significant means any conflict outside `TODO.md`.
4. A branch whose changes are fully contained in `main` is not a conflict;
   report it as a landed/retire candidate instead of rebasing.
5. Remove any temporary worktree it created, then report: rebased (old→new
   head) | aborted (conflicted files) | retire candidate | skipped (reason).

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
   remaining worktree, skip anything with an open task doc not yet retired,
   and skip any branch tied to an outstanding (not landed/done) `TODO.md`
   item — a merged branch with a still-open item is a backlog inconsistency;
   report it, do not delete.
4. Remote prune: `git remote prune origin` drops stale tracking refs (safe).
5. Fully merged remote branches: from `git branch -r --merged main`, delete
   with `git push origin --delete <branch>`. Never `main`/`master`, never a
   branch whose local counterpart survived step 3. This is the only permitted
   push; the merged check gates it.

## After

1. No cross-branch `TODO.md` equality pass: the `merge=ours` driver already
   reconciled each rebased branch to `main`. Confirm the driver is installed
   (`git config --local --get merge.ours.driver` = `true`, else
   `uv run cxr-dev bootstrap`) and that `main:TODO.md` stayed accurate.
2. Do not run test suites per branch; dispatch and workers own validation.
3. Report a table: branch → rebased old→new | skipped (reason) | aborted
   (conflict files) | retire candidate; plus chores done per item (worktree
   removed, branch deleted, remote pruned/deleted) with recovery SHAs; plus
   backlog findings (merged-but-still-outstanding TODO items, active branches
   missing from `TODO.md` as triage candidates). List branches now diverged
   from their upstream (each needs `--force-with-lease`) and stop; push only
   on explicit user instruction.
