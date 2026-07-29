---
name: dispatch-task
description: Use when a user or supervisor agent needs to assign cxr-mc backlog implementation to another agent, select an authority tier, resolve its task branch/worktree, or produce a complete worker handoff.
---

# Dispatch Task

Route one owned implementation slice. Do not perform implementation unless
caller also assigns it to this agent.

## Inventory

1. Invoke `todo-sync`; read `main:TODO.md`, matching `tasks/<branch-leaf>.md`,
   `git worktree list --porcelain`, target branch status, and relevant
   instructions.
2. Use `repo-orientation` and Tokensave to confirm owners, dependencies, and
   affected tests. Exact paths and non-code text may use `rg` or direct reads.
3. Prefer existing linked worktree. For new tasks, let one designated writer
   create/sync task doc, TODO entry, branch, and worktree. Complete required
   setup commit/push before dispatch.
4. Stop on unrelated dirty state, missing/inaccessible worktree, ambiguous
   backlog ownership, or TODO divergence.

## Select worker

Risk and scope override model price/name.

| Skill | Example tier | Assign |
|---|---|---|
| `implement-task-lite` | Haiku, Luna | Mechanical, specified owner, one small acceptance criterion |
| `implement-task` | Sonnet, Terra | Coherent checklist slice, known design, bounded cross-file work |
| `lead-task` | Opus, Sol, Fable, K3 | Whole milestone, ambiguity, physics, integration, or delegation |

Do not split by CLI/UI/physics/performance/docs. Add matching domain skills to
handoff: `cli-ui-ux`, `notebook-workflow`, `physics-review`,
`physics-validation`, `performance`, `documentation-maintenance`,
`remote-gpu-jobs`, `regression-testing`, or `scientific-library`.

## Dispatch contract

Send explicit values; never rely on child inheriting supervisor context.

```text
Worker skill:
Task:
Task file:
Worktree:
Branch:
Slice / non-goals:
Done when:
Required skills:
Expected owners/tests:
Authority:
  commits: checkpoint | final | no
  push: yes | no
  todo-writer: yes | no
  delegate: yes | no
Stop conditions:
Report:
```

Direct user invocation defaults to `commits: checkpoint`, `push: no`,
`todo-writer: no`, `delegate: no`. A supervisor may pass only authority it
already has. Use one writer per worktree; parallelize read-only investigation
or isolated branches, never overlapping edits.

If agent dispatch is unavailable, return exact handoff prompt and target skill.
