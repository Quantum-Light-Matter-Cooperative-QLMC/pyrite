---
name: dispatch-task
description: Use when assigning cxr-mc backlog implementation, completing approved task branch/worktree setup, tearing down a landed task, selecting a worker tier, or producing a complete handoff.
---

# Dispatch Task

Route one owned implementation slice. Do not perform implementation unless
caller also assigns it to this agent.

## Inventory

1. Read `main:TODO.md`, matching `tasks/<branch-leaf>.md`,
   `git worktree list --porcelain`, branch status, and relevant instructions.
2. Use `repo-orientation` and Tokensave to confirm owners, dependencies, and
   affected tests. Exact paths and non-code text may use `rg` or direct reads.
3. If item still contains `>user<`, invoke `triage`; stop for plan review.
4. Stop on unrelated dirty state, missing/inaccessible worktree, ambiguous
   backlog ownership, or TODO divergence.

## Lifecycle

### Start approved task

1. Verify reviewed task doc, one-line TODO pointer, branch, and worktree.
2. Create/reuse missing branch/worktree; keep one TODO writer.
3. Verify triage's setup commit (task doc + synchronized `TODO.md`) exists and
   `main` and the task branch are pushed with upstream.
4. Dispatch only after clean status and remote setup verification.

Direct user invocation authorizes TODO ownership. Setup commit and setup push
belong to `triage`; do not grant worker push or TODO authority implicitly.

### Retire landed task

Only when explicitly asked:

1. Verify branch landed and identify durable task content.
2. Promote durable content to `docs/`; remove task file; update backlog.
3. Commit and push authorized lifecycle changes.
4. Remove worktree/branch only after clean-state and recovery checks. Report
   deleted local/remote refs.

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

Worker handoff from direct user invocation defaults to `commits: checkpoint`,
`push: no`, `todo-writer: no`, `delegate: no`. A supervisor may pass only
authority it already has. Use one writer per worktree; parallelize read-only
investigation or isolated branches, never overlapping edits.

If agent dispatch is unavailable, return exact handoff prompt and target skill.
