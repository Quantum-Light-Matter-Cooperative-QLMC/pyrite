# `tasks/` — branch-scoped working docs

One directory per in-progress branch: `tasks/<branch-name>/`, keyed by the full
task branch name (slashes nest, e.g. `tasks/feature/oom-stage2/`). The entry doc
is `tasks/<branch-name>/README.md`; add supporting docs beside it in the same
directory. Store problem, implementation path, checklist, decisions, and
delegation plan here.

Directory-per-full-branch-name (not a flat `<branch-leaf>.md`) keeps each task
set's files grouped and uniquely pathed. If `main` fast-forwards with multiple
task sets still present, they stay in separate directories instead of colliding
on a shared leaf.

## Invariant

- `main:TODO.md` is the single source of truth: one summary per item plus a
  branch/task pointer. Branch copies are disposable and need not match.
- Branch detail never enters `TODO.md` or durable `docs/`.
- Merge/rebase never prompts for TODO.md conflict resolution. The
  `.gitattributes` `TODO.md merge=ours` driver resolves conflicting hunks to the
  current branch (main's copy when a task branch merges in, or the rebase base).
  The driver lives in local git config — run `uv run cxr-dev bootstrap` once per
  clone; without it, git falls back to a 3-way merge and conflicts return.
- Always drop a completed task's line on `main` (`triage` / `dispatch-task`
  retire) — the authoritative writer. The driver keeps main's copy for any hunk
  main also touched, and adjacent list lines merge into one hunk, so a
  branch-side removal is only honoured when main never edited that region.
  Don't rely on it; drop on `main`.

## Workflow

1. Use `triage` on existing `>user<` prose or invoke `/triage <text>` directly:
   read linked design, split independently ownable tasks, draft each
   `tasks/<branch-name>/README.md`, create local branches/worktrees, and stop
   for review.
2. Replace reviewed prose with one-line `TODO.md` summary:
   `→ feature/<branch>; tasks/<branch-name>/`.
3. Use `todo-sync` only to check `main:TODO.md` accuracy and confirm the merge
   driver is installed — not to force branch equality.
4. Use `dispatch-task` to commit approved setup, push branch with upstream,
   then assign an explicit slice.
   Choose `implement-task-lite`, `implement-task`, or `lead-task` by task
   scope/risk; model label is secondary. State acceptance checks, required
   domain skills, commit/push/TODO/delegation authority, and stop conditions.
   Portable prompt: `Use the dispatch-task skill for <task>`. Clients may also
   expose `$dispatch-task` or `/dispatch-task`.
5. Commit independently valid checkpoints when authorized: focused checks
   pass, scoped diff reviewed, explicit paths staged. Never mix unrelated WIP.
6. Ask `dispatch-task` to retire landed work: promote durable content to
   `docs/`, remove the task directory, update/sync backlog, then remove
   branch/worktree only after verification.
