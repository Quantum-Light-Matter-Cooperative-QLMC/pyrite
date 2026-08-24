---
name: implement-task-lite
description: Use when implementing one tightly bounded, low-risk PyRITE issue slice with explicit owners and acceptance checks.
---

# Implement Task Lite

Execute only the assigned slice. No architecture work, broad refactor,
delegation, physics changes, or backlog management.

1. Confirm issue/handoff, cwd, branch, worktree, clean-enough status, and
   authority. Read the named owner files/tests and any required domain skill.
2. If ownership is not explicit, use `repo-orientation` once. Use Serena only
   when symbol references/callers are needed; do not spend actions activating,
   reading manuals, or inspecting tool config unless the tool actually fails.
3. Make the smallest complete change. Add a focused regression test when
   behavior changes; do not widen cleanup.
4. Run the named/focused checks. Do not run heavy GPU sweeps locally.
5. If `commits` permits, inspect the scoped diff, stage explicit task paths, and
   commit one logical result. Never `git add .`; never commit known broken state.
6. Do not edit the GitHub issue unless `issue-writer: yes`; if authorized, update
   only the checklist/state materially changed by this slice. Do not push or
   open a PR without the corresponding authority.
7. Report changed paths, checks, commit hash(es), ahead count, and exact
   remainder/blocker.

Stop on overlapping edits, a required design decision, physics/scope expansion,
or failure of a prerequisite needed for the assigned slice.
