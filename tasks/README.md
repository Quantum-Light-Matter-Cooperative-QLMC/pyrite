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

- `TODO.md` equals `main:TODO.md` on every branch: one summary per item plus
  branch/task pointer.
- Branch detail never enters `TODO.md` or durable `docs/`.
- This avoids silent fast-forward backlog replacement.

## Workflow

1. Use `triage` on existing `>user<` prose or invoke `/triage <text>` directly:
   read linked design, split independently ownable tasks, draft each
   `tasks/<branch-name>/README.md`, create local branches/worktrees, and stop
   for review.
2. Replace reviewed prose with one-line `TODO.md` summary:
   `→ feature/<branch>; tasks/<branch-name>/`.
3. Use `todo-sync` only to keep branch and main `TODO.md` identical.
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
