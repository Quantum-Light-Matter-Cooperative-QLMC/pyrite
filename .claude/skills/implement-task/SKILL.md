---
name: implement-task
description: Use when implementing a coherent cxr-mc task checklist slice on its existing task branch/worktree with bounded independent investigation, tests, and checkpoint commits.
---

# Implement Task

## Start

1. Confirm handoff, cwd, branch, worktree, status, merge state, and authority.
   Read `AGENTS.md`, `main:TODO.md`, matching task file, and required skills.
2. Invoke `todo-sync` and verify TODO invariant. Do not edit TODO unless
   `todo-writer: yes`; preserve `>user<` text.
3. Invoke `repo-orientation`; read `docs/repo_map.md`, then use Tokensave for
   owners, helpers, callers/callees, impact, and affected tests. Use Context7
   only for current external-library docs.
4. Convert assigned slice into small acceptance checks. Respect task decisions
   and non-goals; stop for material design ambiguity.

## Implement

1. Prefer failing regression test when it proves behavior. Edit smallest owning
   surface; keep reusable logic in `src/cxr_mc/` and apps thin.
2. Invoke matching domain skills for CLI, notebooks, physics, performance,
   docs, remote compute, scientific code, or stochastic tests.
3. Verify each coherent slice with smallest useful fresh check. Use real CLI or
   app probe where static tests cannot prove behavior. Never run heavy GPU work
   locally.
4. When a slice is independently valid and `commits` permits:
   - inspect status and scoped diff;
   - stage explicit assigned paths only; never `git add .`;
   - commit logical checkpoint with tests passing;
   - record hash in task doc when useful.
5. Commit before risky next phase or handoff, not by clock. Never commit known
   broken state merely to save progress. Never amend/rewrite another worker's
   commits. Never push unless `push: yes`.
6. Update assigned task checklist/decisions/remainder. Keep `TODO.md` summary
   only; durable landed design belongs in `docs/`.

## Finish

Run `verifying-changes`; inspect final scoped diff and branch ahead count.
Report results, checks, commits, remaining checklist, and blockers. Leave
worktree on assigned branch. Stop on overlapping dirty work, TODO divergence,
new authority need, or scope outside handoff.
