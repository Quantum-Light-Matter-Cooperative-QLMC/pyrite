---
name: triage
description: Use when /triage is invoked with optional task-description text or user-marked backlog prose must become reviewable, committed, and pushed cxr-mc task plans, task branches/worktrees, and canonical TODO pointers before dispatch.
---

# Triage

Convert task prose into reviewable, published work. Judgment plus setup commit
and branch push; stop before dispatch.

## Input

- `/triage`: select existing `>user<` entries from `main:TODO.md`.
- `/triage <text>`: treat arguments as fresh `>user<` prose. Process only that
  text, not unrelated existing markers. Split into multiple tasks only when
  independently ownable; preserve all supplied intent.

## Workflow

1. Read authoritative `main:TODO.md`. Resolve input per above. Inspect branches,
   `git worktree list --porcelain`, and statuses. Do not retouch tasks already
   owned by a branch.
2. For each task, preserve full intent and read every linked design doc. Use
   `repo-orientation` and Tokensave when implementation ownership needs
   confirmation.
3. For each task, derive `<branch-name>` (the full task branch name) and draft
   `tasks/<branch-name>/README.md` (slashes nest, e.g.
   `tasks/feature/oom-stage2/README.md`) with:

   - problem and scope
   - implementation path and likely owners
   - stepwise checklist
   - decisions/open questions
   - delegation slices and required skills
   - acceptance checks

4. Create each local task branch/worktree if absent. Stop on name collision,
   unrelated dirty state, ambiguous intent, or missing design evidence.
5. For each task, replace its existing marker or insert direct-input work as one
   canonical `TODO.md` summary with branch and `tasks/<branch-name>/` pointers.
   Place it in the appropriate priority/state section. Invoke `todo-sync`.
6. Commit the setup on `main` in one commit: stage explicit paths only
   (`TODO.md`, each new `tasks/<branch-name>/`); never `git add .`, never sweep
   unrelated dirty state. Use a `docs(tasks): triage <branch-name>`-style
   message.
7. Push `main` and each new task branch with upstream (`git push -u`). Verify
   remote refs before presenting.
8. Show task docs, TODO diff, commit, pushed branches/worktrees, assumptions,
   and open decisions for user review. Address feedback by amending the setup
   commit and re-pushing.

Do not implement or dispatch. After approval, `dispatch-task` owns worker
handoff and eventual teardown.

