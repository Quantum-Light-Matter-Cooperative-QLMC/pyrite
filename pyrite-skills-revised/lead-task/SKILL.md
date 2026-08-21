---
name: lead-task
description: Use when a high-capability agent must own, decompose, implement, and integrate a complex PyRITE GitHub issue milestone, including ambiguous, physics, performance, or delegated work.
---

# Lead Task

Own the issue outcome, not unlimited repository scope.

## Establish control

1. Read the issue, handoff authority, branch/worktree state, dependencies, and
   acceptance criteria. Check only worktrees/branches relevant to the milestone.
2. Resolve goal, non-goals, decisions, owners, dependency order, integration
   points, and risks. Use `repo-orientation` for cross-cutting ownership and
   Serena when symbol-level navigation materially helps; do not perform tool
   setup ceremony by default.
3. Invoke only matching domain skills. Require fresh-context physics validation
   for changed physics; route heavy compute through `remote-gpu-jobs`.
4. If a long working plan is useful, keep it as disposable branch-local
   `agentdocs/` scratch. The GitHub issue remains canonical; durable landed
   decisions belong in docs/source/tests.

## Execute

- Decompose into independently verifiable milestones. Delegate only with
  `delegate: yes`, disjoint ownership, an explicit worktree/branch, worker
  skill, acceptance checks, authority, and report contract.
- One writer per mutable worktree. Parallel workers edit isolated branches or
  perform read-only investigation.
- Implement/integrate the smallest owners and verify before checkpoints.
- If `commits` permits, commit logical valid milestones: inspect scoped diff,
  stage explicit paths, run focused checks, commit. Never `git add .`, commit
  broken state, or rewrite another worker's commits.
- Reconcile delegated results from diffs/tests, not summaries alone. Push only
  with `push: yes`.
- With `issue-writer: yes`, update issue checkboxes/decisions at meaningful
  milestones only. With `pr: yes`, create/update the PR and use `Closes #<n>`
  when merge should retire the issue.

## Close milestone

Run focused, neighboring/runtime, then broader checks proportional to risk.
Inspect the integrated diff and reconcile delegated artifacts. Report
branch/worktree, ahead count, checks, risks, completed acceptance items, and the
next remaining slice. Do not claim landing until integration is confirmed.
