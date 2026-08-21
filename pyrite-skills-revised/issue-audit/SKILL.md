---
name: issue-audit
description: Use only for a read-only audit of PyRITE GitHub Issue backlog hygiene, ownership, dependencies, and linked development branches.
---

# Issue Audit

GitHub Issues are the single backlog/task-plan source of truth. This skill is
for explicit backlog audits, not a prerequisite for implementation.

1. List open issues with structured output, e.g.
   `gh issue list --state open --limit 200 --json number,title,labels,assignees,updatedAt,url`.
   Add server-side filters when the audit is scoped.
2. Flag clear hygiene problems: duplicate/obsolete issues, missing ownership or
   useful labels, stale blocked state, and issues whose stated plan/acceptance
   criteria are not actionable.
3. For issues that claim active development, inspect linked branches with
   `gh issue develop --list <n>` only as needed; do not query every issue when
   there is no reason.
4. Report findings without editing issues, branches, or worktrees.

Optional `agentdocs/` files are noncanonical scratch and are outside backlog
consistency checks.
