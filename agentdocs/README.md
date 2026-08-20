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
Retire landed task directories once the GitHub issue and durable docs are correct.

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

- Open GitHub Issues are the single source of truth: one issue per item,
  labelled `priority:*`/`status:*`/`area:*`, with a branch/task pointer in the
  body where one exists. Branch `agentdocs/tasks/` copies are disposable and
  need not match the issue verbatim.
- Branch detail never enters an issue body or durable `docs/`.
- Always have `dispatch-task` close a completed task's issue when retiring
  landed work — the authoritative writer (`gh issue close`, with a comment
  pointing at the landing commit/PR).

## Workflow

1. Use `triage` on existing `status:needs-triage` issues, `>user<` prose, or
   invoke `/triage <text>` directly: read linked design, split independently
   ownable tasks, draft each `agentdocs/tasks/<branch-name>/README.md`, open or
   relabel the canonical GitHub issue (priority/area labels, branch/task-doc
   pointer in the body), and commit the task-doc setup on `main`.
2. Create each new branch/worktree from the setup commit, push `main` and the
   task branches, then stop for review. Address review through `triage` with a
   follow-up setup commit; advance each still-unstarted task branch to the
   reviewed commit before dispatch.
3. Use `todo-sync` only to audit open-issue/task-doc consistency (accurate
   branch/task pointers, no orphaned `agentdocs/tasks/` directories) — not to
   force branch equality.
4. Use `dispatch-task` to verify approved/pushed setup, then assign an explicit
   slice.
   Choose `implement-task-lite`, `implement-task`, or `lead-task` by task
   scope/risk; model label is secondary. State acceptance checks, required
   domain skills, commit/push/issue/delegation authority, and stop conditions.
   Portable prompt: `Use the dispatch-task skill for <task>`. Clients may also
   expose `$dispatch-task` or `/dispatch-task`.
5. Commit independently valid checkpoints when authorized: focused checks
   pass, scoped diff reviewed, explicit paths staged. Never mix unrelated WIP.
6. Ask `dispatch-task` to retire landed work: promote durable content to its
   owning artifact (`README.md`, `docs/`, ADR, source documentation, or tests),
   remove the task directory, close the issue, then remove branch/worktree
   only after verification.
