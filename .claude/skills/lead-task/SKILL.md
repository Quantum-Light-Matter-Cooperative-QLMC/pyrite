---
name: lead-task
description: Use when a high-capability agent must own, decompose, implement, and integrate a complex cxr-mc task milestone, including ambiguous or cross-subsystem work, physics, performance, or supervised delegation.
---

# Lead Task

Own milestone outcome, not unlimited repository scope.

## Establish control

1. Read `AGENTS.md`; invoke `todo-sync` and `repo-orientation`. Confirm
   `main:TODO.md`, task doc (`tasks/<branch-name>/README.md`), all
   worktrees/branches, clean ownership, merge state, dependencies, and authority.
   Activate Serena, read its manual, and inspect current config. If requested
   `one-shot` is absent on a shared server, retain the handoff's autonomous
   contract without changing project-wide modes.
2. If authorized as TODO writer, preserve invariant and `>user<` text. New-task
   setup requires task doc + synced TODO setup commit and upstream push before
   implementation. Otherwise never edit TODO.
3. Resolve goal, non-goals, decisions, acceptance evidence, owning paths,
   dependency order, integration points, and risks. Record branch-specific
   detail in the task doc, durable landed decisions in `docs/`.
4. Follow `repo-orientation` for navigation and invoke every matching domain
   skill. Require fresh-context `physics-validation` for changed physics; route
   heavy compute through `remote-gpu-jobs`.

## Execute

- Decompose into independently valid milestones. Delegate only with
  `delegate: yes`, disjoint ownership, explicit worktree/branch, worker skill,
  acceptance checks, authority, and report contract.
- One writer owns TODO and each mutable worktree. Parallel workers use isolated
  branches/worktrees; read-only investigation may share context.
- Implement or integrate smallest owners. Verify before checkpoint.
- If `commits` permits, commit every independently valid milestone and before
  risky phases/handoffs: inspect status/scoped diff, stage explicit paths, run
  focused checks, commit. Never `git add .`, commit broken state, rewrite other
  workers' commits, or mix unrelated WIP.
- Reconcile worker results from diffs/tests, not summaries alone. Resolve
  integration conflicts deliberately. Push only with `push: yes`.

## Close milestone

Run focused, neighboring, runtime, then broader checks proportional to risk.
Inspect the integrated diff and reconcile every delegated result from artifacts.
Update task checklist, decisions, remainder, commits, and evidence. Report
branch/worktree, ahead count, checks, risks, and next slice. Do not mark work
landed or retire task state without confirmed integration and dispatch authority.
