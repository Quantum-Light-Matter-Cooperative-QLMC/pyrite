# `tasks/` — branch-scoped working docs

One file per in-progress branch: `tasks/<branch-leaf>.md`. Store problem,
implementation path, checklist, decisions, and delegation plan here.

## Invariant

- `TODO.md` equals `main:TODO.md` on every branch: one summary per item plus
  branch/task pointer.
- Branch detail never enters `TODO.md` or durable `docs/`.
- This avoids silent fast-forward backlog replacement.

## Workflow

1. Create task branch and `tasks/<branch-leaf>.md`.
2. Add one-line `TODO.md` summary:
   `→ feature/<branch>; tasks/<branch-leaf>.md`.
3. Keep branch and main `TODO.md` identical; use `todo-sync`.
4. Before dropping landed branch, promote durable content to `docs/`, remove
   task file, and update backlog state.
