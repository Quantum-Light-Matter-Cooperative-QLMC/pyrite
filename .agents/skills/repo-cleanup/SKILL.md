---
name: repo-cleanup
description: "Use when cleaning PyRITE git/worktree: requested safe rebases, stale tracking cleanup, and removal of objectively merged worktrees/refs; never edit GitHub issues."
---
# Repo Cleanup

Own git objects and worktrees only. Never edit issues or implementation plans.
Never force-delete or force-push unless the user explicitly authorizes that
exact operation.

## Inventory

1. Fetch/prune, then inspect relevant branches/worktrees, dirty/merge state,
   ahead/behind counts, and merge bases against `main`.
2. For a deletion candidate, verify it is merged into `main`, its worktree is
   clean, and no open PR targets it. If its name identifies an issue (for
   example `issue-123-*`), check that issue; an open issue is a reason to skip.
3. Skip unclear ownership, dirty worktrees, active merge/rebase state, or any
   branch not objectively safe for the requested operation.

## Rebase requested branches

Do not fan out subagents by default. For each requested branch:

1. Preflight conflict risk (`git merge-tree --write-tree main <branch>` when
   appropriate).
2. Rebase only when the target worktree is clean and ownership is clear.
3. Abort on conflict unless conflict resolution was explicitly assigned; report
   old/new head or abort reason.
4. Never switch branches inside an unrelated existing checkout; use its own or a
   temporary worktree.

## Remove merged state

1. Record the branch tip SHA first.
2. Remove a clean merged worktree without `--force`.
3. Delete the fully merged local branch with `git branch -d`.
4. Prune stale remote-tracking refs.
5. Delete a merged remote branch only with explicit authority and only after
   local cleanup; never delete `main`/`master`.

Do not run task test suites. Report each requested branch: action, old/new or
recovery SHA, skipped reason, and any upstream divergence requiring separate
`--force-with-lease` approval.
