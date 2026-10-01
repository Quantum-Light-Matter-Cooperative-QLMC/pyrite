---
name: implement-task-lite
description: Use when implementing one tightly bounded, low-risk PyRITE issue slice with explicit owners and acceptance checks.
---

# Implement Task Lite

Execute only the assigned slice. No architecture work, broad refactor,
delegation, physics changes, or backlog management.

1. Confirm issue/handoff, cwd, branch, worktree, clean-enough status, and
   authority. Confirm the branch is not already merged and contains current
   `origin/main` (`git merge-base --is-ancestor origin/main HEAD`); if behind,
   stop and report rather than rebasing. No handoff (direct user invocation)
   means `commits: checkpoint`; `push`, `issue-writer`, `pr`, and `delegate`
   are `no`.
2. Read the assigned issue Plan/Acceptance items, named owner files/tests, and
   any required domain skill.
3. If ownership is not explicit, use `repo-orientation` once. Use Serena only
   when symbol references/callers are needed; do not spend actions activating,
   reading manuals, or inspecting tool config unless the tool actually fails.
4. Make the smallest complete change. Add a focused regression test when
   behavior changes; do not widen cleanup.
5. Run the named/focused checks. Do not run heavy GPU sweeps locally.
6. If `commits` permits, inspect the scoped diff, stage explicit task paths, and
   commit one logical result. Never `git add .`; never commit known broken state.
7. Do not edit the GitHub issue unless `issue-writer: yes`; if authorized, tick
   only the boxes this slice completed. Do not push or open a PR without the
   corresponding authority.
8. Report:
   - changed paths, commits, and ahead count
   - checks run and results
   - issue Plan/Acceptance items completed (quote the box text) and remaining
   - exact blocker, if any

Stop on overlapping edits, a required design decision, physics/scope expansion,
or failure of a prerequisite needed for the assigned slice.
