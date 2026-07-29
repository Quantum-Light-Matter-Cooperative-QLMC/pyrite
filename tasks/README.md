# `tasks/` — branch-scoped working docs

One file per in-progress branch: `tasks/<branch-leaf>.md`. Store problem,
implementation path, checklist, decisions, and delegation plan here.

## Invariant

- `TODO.md` equals `main:TODO.md` on every branch: one summary per item plus
  branch/task pointer.
- Branch detail never enters `TODO.md` or durable `docs/`.
- This avoids silent fast-forward backlog replacement.

## Workflow

1. For `>user<` prose, use `triage`: read linked design, draft
   `tasks/<branch-leaf>.md`, create local branch/worktree, and stop for review.
2. Replace reviewed prose with one-line `TODO.md` summary:
   `→ feature/<branch>; tasks/<branch-leaf>.md`.
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
   `docs/`, remove task file, update/sync backlog, then remove branch/worktree
   only after verification.
