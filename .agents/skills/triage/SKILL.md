---
name: triage
description: Use when /triage is invoked or user-marked backlog prose must become a reviewable cxr-mc task plan, local task branch/worktree, and canonical TODO pointer before dispatch.
---

# Triage

Convert one `>user<` item into a reviewable task. Judgment only; stop before
commit, push, or dispatch.

## Workflow

1. Read authoritative `main:TODO.md`; scan `>user<` markers. Inspect branches,
   `git worktree list --porcelain`, and statuses. Do not retouch items already
   owned by a branch.
2. Select one marker. Preserve its full intent; read every linked design doc.
   Use `repo-orientation` and Tokensave when implementation ownership needs
   confirmation.
3. Derive `<leaf>` and draft `tasks/<leaf>.md` with:

   - problem and scope
   - implementation path and likely owners
   - stepwise checklist
   - decisions/open questions
   - delegation slices and required skills
   - acceptance checks

4. Create local task branch/worktree if absent. Stop on name collision,
   unrelated dirty state, ambiguous intent, or missing design evidence.
5. Replace source `>user<` prose with one-line summary, branch pointer, and
   `tasks/<leaf>.md` pointer. Invoke `todo-sync` to restore exact TODO equality.
6. Show task doc, TODO diff, branch/worktree, assumptions, and open decisions
   for user review.

Do not commit, push, implement, or dispatch. After approval, `dispatch-task`
owns setup commit, upstream push, worker handoff, and eventual teardown.
