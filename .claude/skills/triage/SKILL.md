---
name: triage
description: Use when new task prose or an existing GitHub issue must become a reviewable, canonical PyRITE issue plan before implementation or dispatch.
---

# Triage

GitHub Issues are the backlog and task-plan source of truth. Triage plans work
and stops for review; it does not implement it, create branches/worktrees or
task docs, commit, or push. `dispatch-task` owns routing to a worker.

## Input

- `/triage <text>`: triage only the supplied task text.
- `/triage #<n>` or an issue URL: refine that issue.
- `/triage` with no target: choose an open issue carrying no `status:*` label
  (or new `>user<` prose) only when selection is unambiguous; otherwise report
  the best candidates.

## Workflow

1. For an existing target, read it with
   `gh issue view <n> --json number,title,body,labels,assignees,state,url`. For
   fresh text, draft the plan first and create the issue once. Preserve user
   intent; split only when pieces are independently ownable and useful to track
   separately.
2. Inspect only enough repository context to make the plan executable. Use
   `repo-orientation` when ownership or impact is unclear; use Serena only when
   symbol-level navigation materially helps.
3. Make the issue body concise and executable. Prefer these sections:

   ```markdown
   ## Goal
   ## Scope
   ## Plan
   - [ ] ...
   ## Acceptance
   - [ ] ...
   ## Decisions / constraints
   ```

   Omit empty sections. Put durable design decisions in their owning docs once
   implemented, not in a second backlog file.
   For new/edited physics claims, each acceptance item names the claim id and
   its target ledger status: `rederived` (derivation/verification work) or
   `anchored` (also pinned by a regression test). Never add a `signed-off`
   item; human sign-off lives only in the standing ledger-review issue #277 (see
   `docs/validation/methodology.md` "Agent done criteria").
4. Use existing `area:*`/type labels when clearly applicable, and apply
   `status:backlog` (accepted, not yet scheduled; `dispatch-task` sets
   `status:active`). Use native issue relations (`--blocked-by`, `--blocking`,
   parent/sub-issue) when they express real dependencies; do not duplicate
   dependency state in prose.
5. Create/update the issue with `gh issue create` or `gh issue edit`. The issue
   body is the only task record; do not create `agentdocs/tasks/` files.
6. Stop for review. Report the issue number/link, key assumptions, unresolved
   material decisions, and whether it is ready for dispatch. Review feedback
   returns through `triage` as an issue-body edit.
