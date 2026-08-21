---
name: dispatch-task
description: Use when an approved GitHub issue needs a linked branch/worktree, worker selection/handoff, or post-merge issue/worktree retirement; excludes implementation.
---

# Dispatch Task

Coordinate one approved GitHub issue. `triage` owns planning; worker skills own
implementation; `repo-cleanup` owns generic git hygiene.

## Start an approved issue

1. Read the issue with `gh issue view <n> --json number,title,body,labels,state,url`
   and confirm it is open, approved, and actionable. Inspect existing linked
   branches with `gh issue develop --list <n>` and current worktrees/status.
2. Reuse an existing clean linked branch/worktree when unambiguous. Otherwise
   create one linked development branch from `main` with `gh issue develop`;
   prefer `issue-<n>-<short-slug>` naming. Fetch it and attach a dedicated
   worktree using the repository's normal worktree convention.
3. Stop on overlapping dirty state in the target worktree, an already-active
   conflicting branch, or a material decision missing from the issue. Unrelated
   dirt in another worktree is not a blocker.
4. Select the smallest worker capable of the approved scope:

| Skill | Assign |
|---|---|
| `implement-task-lite` | One mechanical, low-risk slice with explicit owner/check |
| `implement-task` | Bounded cross-file checklist slice with known design |
| `lead-task` | Ambiguous/cross-subsystem milestone, physics, performance, integration, or delegation |

Add only domain skills that the slice actually needs. Do not force
`repo-orientation`, Serena, or other setup tools when the issue/handoff already
identifies owners and checks.

## Handoff

Send explicit context; do not rely on inherited supervisor state.

```text
Worker skill:
Issue: #<n> <url>
Worktree:
Branch:
Slice / non-goals:
Done when:
Required skills:
Expected owners/tests:
Authority:
  commits: checkpoint | final | no
  push: yes | no
  issue-writer: yes | no
  pr: yes | no
  delegate: yes | no
Stop conditions:
Report:
```

Default direct-user authority: `commits: checkpoint`, `push: no`,
`issue-writer: no`, `pr: no`, `delegate: no`. One writer per mutable worktree;
parallel edits require isolated branches/worktrees. If worker dispatch is not
available, return this filled handoff and target skill instead.

## After landing

When explicitly asked to retire completed work:

1. Verify the change landed on `main` and identify its PR/commit.
2. Prefer a PR containing `Closes #<n>` so GitHub closes the issue on merge. If
   the change landed without automatic closure, close the issue with a short
   landing reference.
3. Ensure durable decisions/documentation live in their owning artifacts. Do not
   preserve disposable agent scratch merely for history.
4. Hand safe worktree/local-ref removal to `repo-cleanup` and report recovery
   SHAs.
