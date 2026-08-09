# Agent work records

This is the sole tracked home for agent-authored implementation plans,
branch-task records, handoffs, and reports. It is intentionally outside
`docs/`, which contains durable project documentation and may be published.
Use ignored `scratch/` or `/tmp` for disposable notes; do not create parallel
`docs/plans`, `docs/temp`, root `tasks`, `.remember`, or untracked `agentdocs`
surfaces.

- `tasks/<branch-name>/` — current branch-scoped work; entry doc `README.md`.
- `plans/` — cross-task sequencing that cannot belong to one branch.
- `archive/` — retained historical handoffs only; never an active authority.

Promote durable outcomes to `README.md`, `docs/`, an ADR, source documentation,
or tests.
Retire landed task directories once `TODO.md` and durable docs are correct.

## Branch task records

One directory per in-progress branch: `agentdocs/tasks/<branch-name>/`, keyed by
the full task branch name (slashes nest, e.g.
`agentdocs/tasks/feature/oom-stage2/`). The entry doc is
`agentdocs/tasks/<branch-name>/README.md`; add supporting docs beside it. Store
problem, implementation path, checklist, decisions, and delegation plan here.

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
- Always have `dispatch-task` drop a completed task's line on `main` when
  retiring landed work — the authoritative writer. The driver keeps main's copy for any hunk
  main also touched, and adjacent list lines merge into one hunk, so a
  branch-side removal is only honoured when main never edited that region.
  Don't rely on it; drop on `main`.

## Workflow

1. Use `triage` on existing `>user<` prose or invoke `/triage <text>` directly:
   read linked design, split independently ownable tasks, draft each
   `agentdocs/tasks/<branch-name>/README.md`, replace prose with a one-line
   `TODO.md` pointer, and commit that setup on `main`.
2. Create each new branch/worktree from the setup commit, push `main` and the
   task branches, then stop for review. Address review through `triage` with a
   follow-up setup commit; advance each still-unstarted task branch to the
   reviewed commit before dispatch.
3. Use `todo-sync` only to check `main:TODO.md` accuracy and confirm the merge
   driver is installed — not to force branch equality.
4. Use `dispatch-task` to verify approved/pushed setup, then assign an explicit
   slice.
   Choose `implement-task-lite`, `implement-task`, or `lead-task` by task
   scope/risk; model label is secondary. State acceptance checks, required
   domain skills, commit/push/TODO/delegation authority, and stop conditions.
   Portable prompt: `Use the dispatch-task skill for <task>`. Clients may also
   expose `$dispatch-task` or `/dispatch-task`.
5. Commit independently valid checkpoints when authorized: focused checks
   pass, scoped diff reviewed, explicit paths staged. Never mix unrelated WIP.
6. Ask `dispatch-task` to retire landed work: promote durable content to its
   owning artifact (`README.md`, `docs/`, ADR, source documentation, or tests),
   remove the task directory, update/sync backlog, then remove branch/worktree
   only after verification.
