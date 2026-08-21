---
name: todo-sync
description: Use when open-GitHub-Issue backlog accuracy needs a read-only audit; never edit issues or task files.
---

# Backlog Sync

Open GitHub Issues are the single source of truth. Branch
`agentdocs/tasks/<branch-name>/` copies are disposable and need not match an
issue verbatim.

1. `gh issue list --state open --limit 200` (add `--label` filters as needed).
   Verify each issue tied to in-flight work carries an accurate branch and
   `agentdocs/tasks/<branch-name>/` pointer in its body.
2. Cross-check `agentdocs/tasks/` directories against open issues: flag a task
   directory with no matching open issue, or an issue pointer naming a
   directory that doesn't exist.
3. Report stale/missing pointers and orphaned task directories without editing.
   `triage` owns new pointers; `dispatch-task` retirement owns closing
   completed-item issues.

Do not open, close, edit, or label any issue; do not edit task files.
