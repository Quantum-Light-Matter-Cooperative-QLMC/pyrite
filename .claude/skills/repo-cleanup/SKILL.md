---
name: repo-cleanup
description: "Use when cxr-mc needs general git hygiene after task records are correct: safe branch rebases, stale tracking cleanup, and objectively merged worktree/ref removal; never edit TODO or task docs."
---

# Repo Cleanup

Own git objects and worktrees only. `triage` creates task records;
`dispatch-task` retires TODO/task records before task-specific cleanup. Never
edit those records. Never push except an explicitly requested, fully merged
remote-branch deletion.

## Inventory

1. Fetch/prune, then inspect branches, worktrees, dirty/merge state,
   ahead/behind counts, and merge bases against `main`.
2. Read `main:TODO.md` without editing. Active untracked branches are triage
   candidates, never deletion candidates.
3. Skip branches not behind main, dirty worktrees, active merge/rebase state,
   open task records, or unclear ownership.

## Rebase per branch

Fan out one cheap subagent per candidate. One agent per branch/worktree; explicit
authority: preflight and rebase one branch, resolve only `TODO.md`, no push,
force, or tests. Never switch branches in an existing checkout; use a fresh
temporary worktree when a checkout is required.

1. Preflight with `git merge-tree --write-tree main <branch>`.
2. Rebase only with no conflicts or `TODO.md`-only conflicts. The installed
   `merge=ours` driver should resolve TODO; run `uv run cxr-dev bootstrap` if
   missing.
3. Abort on every conflict outside `TODO.md`; do not resolve it.
4. Report branches already contained in main as retirement candidates.
5. Remove temporary worktrees; report old/new heads or abort reason.

## Cleanup chores

Run after rebases. Delegate one named deletion target per cheap subagent. Record
tip SHA first; never use force, wildcard/bulk deletion, or touch main.

1. Prune stale worktree administration.
2. Remove a worktree only when clean and its branch is merged or gone; never
   use `--force`.
3. Delete a fully merged local with `git branch -d` only after its worktree and
   task/TODO records are gone.
4. Prune stale remote-tracking refs.
5. Delete a fully merged remote only with explicit authority, after its local
   counterpart is gone; never delete main/master.

## After

Confirm the TODO merge driver remains installed; do not compare branch TODO
copies. Do not run task test suites. Report per branch: rebased old→new,
skipped/aborted reason, retirement candidate, removed worktree/ref with recovery
SHA, backlog inconsistency, and upstream divergence requiring later explicit
`--force-with-lease` authority.
