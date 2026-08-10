---
name: triage
description: Use when /triage is invoked with optional task-description text or user-marked backlog prose must become reviewable, tracked PyRITE agent task records, task branches/worktrees, and canonical TODO pointers before dispatch.
---

# Triage

Convert task prose into reviewable, tracked work. Judgment plus setup commit
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
   `repo-orientation` and Serena when implementation ownership needs
   confirmation.
3. For each task, derive `<branch-name>` (the full task branch name) and draft
   `agentdocs/tasks/<branch-name>/README.md` (slashes nest, e.g.
   `agentdocs/tasks/feature/oom-stage2/README.md`) with:

   - problem and scope
   - implementation path and likely owners
   - stepwise checklist
   - decisions/open questions
   - delegation slices, required skills, and whether each reviewed slice is
     self-contained enough for Serena `one-shot`
   - acceptance checks

4. For each task, replace its existing marker or insert direct-input work as one
   canonical `TODO.md` summary with branch and
   `agentdocs/tasks/<branch-name>/` pointers.
   Place it in the appropriate priority/state section. Edit only `main:TODO.md`;
   branch copies reconcile via the `merge=ours` driver — no cross-branch sync.
5. Commit the setup on `main` in one commit: stage explicit paths only
   (`TODO.md`, each new `agentdocs/tasks/<branch-name>/`); never `git add .`, never sweep
   unrelated dirty state. Use a `docs(tasks): triage <branch-name>`-style
   message.
6. Create each new local task branch/worktree from that setup commit so the
   worker's branch contains its task record. Stop on name collision, unrelated
   dirty state, ambiguous intent, or missing design evidence; do not retouch a
   branch that already owns work.
7. Push `main` and each new task branch with upstream (`git push -u`). Verify
   remote refs before presenting.
8. Show task docs, TODO diff, commit, pushed branches/worktrees, assumptions,
   and open decisions for user review. Address feedback with a follow-up setup
   commit on `main`, then advance/recreate each still-unstarted task branch from
   the reviewed commit and push it. Stop if implementation has begun; never
   rewrite a worker's branch.

Do not label a slice `one-shot` while a material decision remains open. Do not
implement or dispatch. After approval, `dispatch-task` owns worker launch,
handoff, and eventual task-record retirement.
