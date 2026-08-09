---
name: dispatch-task
description: Use when post-triage work needs approved task setup verification, worker selection/handoff, or landed task-record retirement; excludes implementation and general git cleanup.
---

# Dispatch Task

Coordinate one approved task. `triage` owns initial planning/publication;
worker skills own implementation; `repo-cleanup` owns generic worktree/ref
hygiene.

## Inventory

1. Invoke `todo-sync`. Read the matching
   `agentdocs/tasks/<branch-name>/README.md`,
   worktree list, branch status, and relevant instructions.
2. Invoke `repo-orientation` to confirm owners, dependencies, and tests.
3. If item still contains `>user<`, invoke `triage`; stop for plan review.
4. Stop on unrelated dirty state, missing/inaccessible worktree, ambiguous
   backlog ownership, or TODO divergence.

## Lifecycle

### Start approved task

1. Verify reviewed task doc, one-line TODO pointer, branch, and worktree.
2. Create/reuse the approved worktree if missing; do not redesign the plan.
3. Verify triage's setup commit (task doc + `main:TODO.md` pointer) exists and
   `main` and the task branch are pushed with upstream. Branch `TODO.md` need
   not match main — the `merge=ours` driver reconciles it on merge/rebase.
4. Dispatch only after clean status and remote setup verification.

Setup commit/push belongs to `triage`; do not grant worker push, TODO, or
delegation authority implicitly.

### Retire landed task

Only when explicitly asked:

1. Verify branch landed and identify durable task content.
2. Promote durable content to its owner (`README.md`, `docs/`, ADR, source
   documentation, or tests); remove the task directory; drop the completed
   item's line from main's `TODO.md`.
3. Commit and push authorized lifecycle changes.
4. Hand physical worktree/ref removal to `repo-cleanup`; report its recovery
   SHAs.

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

For an approved, self-contained execution slice, request Serena `one-shot`.
Modes are fixed at MCP startup: when the worker gets its own Serena server,
start it with `--add-mode one-shot`; when workers share a server, put the same
autonomous-completion contract in the handoff and do not edit project defaults.
The mode changes prompting only; it grants no extra authority or approvals.

## Dispatch contract

Send explicit values; never rely on child inheriting supervisor context.

```text
Worker skill:
Task:
Task doc (agentdocs/tasks/<branch-name>/README.md):
Worktree:
Branch:
Slice / non-goals:
Done when:
Required skills:
Expected owners/tests:
Serena execution: one-shot active | one-shot contract (shared server) | interactive
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
