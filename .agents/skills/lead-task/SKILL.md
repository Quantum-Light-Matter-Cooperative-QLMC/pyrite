---
name: lead-task
description: Use when a high-capability agent must own, decompose, implement, and integrate a complex PyRITE GitHub issue milestone, including ambiguous, physics, performance, or delegated work.
---

# Lead Task

Own the issue outcome, not unlimited repository scope.

## Establish control

1. Read the issue body (Goal, Scope, Plan, Acceptance, Decisions), handoff
   authority, branch/worktree state, and dependencies. Check only
   worktrees/branches relevant to the milestone. No handoff (direct user
   invocation) means `commits: checkpoint`; `push`, `issue-writer`, `pr`, and
   `delegate` are `no`.
2. Resolve goal, non-goals, decisions, owners, dependency order, integration
   points, and risks. Use `repo-orientation` for cross-cutting ownership and
   Serena when symbol-level navigation materially helps; do not perform tool
   setup ceremony by default.
3. Invoke only matching domain skills. Require fresh-context physics validation
   for changed physics; route heavy compute through `remote-gpu-jobs`.
4. The issue body is the task record. Refine its `## Plan` only with
   `issue-writer: yes`; otherwise propose plan changes in the report. Keep
   disposable working notes in ignored `scratch/`; use `agentdocs/plans/` only
   for sequencing across issues. Durable landed decisions belong in
   docs/source/tests.

## Execute

- Decompose into independently verifiable milestones mapped to issue Plan
  items.
- Delegate only with `delegate: yes`. Choose the worker as `dispatch-task` does
  (`implement-task-lite` for one mechanical low-risk slice with explicit
  owner/check; `implement-task` for a bounded checklist slice with known
  design) and send its filled handoff template with disjoint ownership, an
  explicit worktree/branch, acceptance checks, and the report contract. Pass
  only authority you hold; never grant `push`, `pr`, `issue-writer`, or
  `delegate` beyond your own.
- One writer per mutable worktree. Parallel workers edit isolated branches or
  perform read-only investigation.
- Implement/integrate the smallest owners and verify before checkpoints.
- If `commits` permits, commit logical valid milestones: inspect scoped diff,
  stage explicit paths, run focused checks, commit. Never `git add .`, commit
  broken state, or rewrite another worker's commits.
- Reconcile delegated results from diffs/tests, not summaries alone. Push only
  with `push: yes`.
- With `issue-writer: yes`, tick verified Plan/Acceptance boxes (including
  delegated ones) and record decisions at meaningful milestones only; otherwise
  never edit the issue. With `pr: yes`, create/update the PR and use
  `Closes #<n>` when merge should retire the issue.

## Close milestone

Run focused, neighboring/runtime, then broader checks proportional to risk.
Inspect the integrated diff and reconcile delegated artifacts. Do not claim
landing until integration is confirmed. Report:

- branch/worktree, commits, and ahead count
- checks run and results
- issue Plan/Acceptance items completed (quote the box text) and remaining
- proposed plan/decision changes, risks, and the next remaining slice
