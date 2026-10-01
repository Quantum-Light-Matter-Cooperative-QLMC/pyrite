# Agent work records

This is the sole tracked home for agent-authored implementation plans,
branch-task records, handoffs, and reports. It is intentionally outside
`docs/`, which contains durable project documentation and may be published.
Use ignored `scratch/` or `/tmp` for disposable notes; do not create parallel
`docs/plans`, `docs/temp`, root `tasks`, `.remember`, or untracked `agentdocs`
surfaces.

- `tasks/` — legacy branch-scoped records; no new ones. The GitHub issue body is the task record.
- `plans/` — cross-task sequencing that cannot belong to one branch.
- `archive/` — retained historical handoffs only; never an active authority.

Promote durable outcomes to `README.md`, `docs/`, an ADR, source documentation,
or tests.
Retire landed task directories once the GitHub issue and durable docs are correct.

## Invariant

- Open GitHub Issues are the single source of truth: one issue per item,
  labelled `status:*`/`area:*`, body holding goal, scope, plan, acceptance, and
  decisions. Branch detail does not enter durable `docs/`.
- `dispatch-task` closes a completed task's issue when retiring landed work
  (`gh issue close`, with a comment pointing at the landing commit/PR).

## Workflow

1. `triage` turns `>user<` prose or unlabelled issues into a canonical issue
   (`status:backlog`) and stops for review.
2. `dispatch-task` creates/reuses the `issue-<n>-<slug>` branch/worktree, sets
   `status:active`, and assigns `implement-task-lite`, `implement-task`, or
   `lead-task` by scope/risk with explicit authority and acceptance checks.
3. Workers commit independently valid checkpoints when authorized: focused
   checks pass, scoped diff reviewed, explicit paths staged.
4. `dispatch-task` retires landed work: confirm durable content lives in its
   owning artifact, close the issue, then `repo-cleanup` removes the
   branch/worktree after verification.
