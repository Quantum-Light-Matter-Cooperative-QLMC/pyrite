---
name: implement-task
description: Use when implementing a coherent, bounded slice of a PyRITE GitHub issue on its linked branch/worktree.
---

# Implement Task

## Start

1. Confirm issue/handoff, cwd, branch/worktree, status/merge state, and
   authority. Read the issue and named owner files/tests; respect its scope,
   decisions, and acceptance criteria.
2. Use `repo-orientation` only when ownership/impact is unclear. Use Serena for
   symbol-level navigation when it is faster than direct search; skip tool
   activation/manual/config ceremony unless troubleshooting the tool itself.
3. Invoke only domain skills materially relevant to the slice. Stop for a
   material design decision not resolved by the issue.

## Implement and verify

1. Prefer a failing regression test when it efficiently proves the behavior.
   Edit the smallest owning surface; keep reusable logic in `src/pyrite/` and
   apps thin.
2. Verify each coherent slice with the smallest useful fresh check, adding a
   real CLI/app probe when static tests cannot prove behavior. Route heavy GPU
   work through `remote-gpu-jobs`; do not run it locally.
3. When a slice is independently valid and `commits` permits: inspect status and
   scoped diff, stage explicit assigned paths, run focused checks, and commit.
   Never `git add .`, commit known broken state, or rewrite another worker's
   commits.
4. Push only with `push: yes`. Open/update a PR only with `pr: yes`; when the PR
   should close the task, include `Closes #<issue>` in the PR body.
5. Edit the issue only with `issue-writer: yes`, and only at meaningful
   milestones (checklist/decision/status), not for routine progress logging.

## Finish

Run focused then neighboring/runtime/broader checks proportional to risk.
Inspect the final scoped diff and ahead count. Report results, checks, commits,
remaining acceptance items, and blockers. Leave the worktree on the assigned
branch.
