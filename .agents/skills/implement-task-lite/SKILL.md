---
name: implement-task-lite
description: Use when implementing one tightly bounded, low-risk cxr-mc task slice with specified owners and acceptance checks, especially through Haiku- or Luna-tier workers.
---

# Implement Task Lite

Execute only supplied slice. No architecture, broad refactor, TODO ownership,
delegation, physics changes, or scope discovery.

1. Confirm cwd, branch, worktree, status, and handoff fields. Read
   `AGENTS.md`, supplied task section, named owners/tests, and required skills.
2. Verify `TODO.md` equals `main:TODO.md`; never edit it. If owner/path is not
   explicit, invoke `repo-orientation` once; stop if still unclear.
3. Use Serena or `rg` for named symbols, impact, and affected tests. Preserve
   unrelated changes.
4. Make smallest complete change. Add focused regression test when behavior
   changes. Do not widen cleanup.
5. Run named/focused checks. Never run heavy sweep locally; use
   `remote-gpu-jobs` only when handoff authorizes remote work.
6. At each independently valid slice, if `commits` permits: inspect status and
   scoped diff, stage explicit task paths only, commit. Never `git add .`.
   Prefer one final commit for short work; never manufacture time-based commits.
7. Update task doc only if handoff names it and no other writer owns it. Never
   push unless `push: yes`.
8. Report outcome, changed paths, checks, commit hashes, ahead count, and exact
   remainder/blocker.

Stop on ambiguity, overlapping dirty files, failing unrelated prerequisites,
required design choice, physics change, or scope exceeding one bounded slice.
