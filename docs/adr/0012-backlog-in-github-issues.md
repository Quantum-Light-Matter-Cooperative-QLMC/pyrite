# 0012 — Track backlog in GitHub Issues, not TODO.md

- **Status:** Accepted
- **Date:** 2026-08-20
- **Supersedes:** ADR-0004's "`TODO.md` is the single backlog authority on `main`" clause

## Context

`TODO.md` held the entire backlog as hand-maintained Markdown: priority sections (Active/P1/P2/P3), an untriaged Inbox, and Long-term plans, each item a summary line plus an optional branch and `agentdocs/tasks/<branch-name>/` pointer. A `.gitattributes` `TODO.md merge=ours` driver (installed via `pyrite-dev bootstrap`) resolved conflicting hunks to the current branch so merge/rebase never prompted for manual TODO.md resolution.

This kept the backlog in-repo and diffable, but had no per-item state beyond prose, no cross-linking, no assignment, and no way to reference an item from outside the repository. The project now tracks issues in the pyrite GitHub repository, which offers native labels, sub-issues, and search that a flat Markdown file cannot.

## Decision

Open GitHub Issues are the single source of truth for backlog items, replacing `TODO.md`. One issue per item, using:

- `priority:p1`/`p2`/`p3`/`long-term` for the former Active/P1/P2/P3/Long-term sections (no `priority:*` label on an untriaged item),
- `status:active`/`gated`/`paused`/`needs-triage` for the former Active/Gated/Paused subsections and untriaged Inbox items,
- `area:*` labels for subsystem (physics, cli, notebooks, docs, tests, performance, remote, geometry, validation, materials, infra, ui),
- the existing bare `bug`/`enhancement`/`documentation`/`question` labels for work-type,
- a branch and `agentdocs/tasks/<branch-name>/` pointer in the issue body where a task record exists.

Nested TODO.md checklists (e.g. per-anchor validation-app failures, a sub-scoped agent-hooks item) became GitHub sub-issues rather than in-body checklists. `TODO.md` is replaced with a short pointer stub; the `.gitattributes` merge driver and `pyrite-dev bootstrap` are removed since there is no longer a file to reconcile per-clone.

`triage`, `dispatch-task`, `todo-sync`, and `repo-cleanup` (see `agentdocs/README.md`) now read/write GitHub Issues instead of `TODO.md` lines; the `agentdocs/tasks/<branch-name>/` task-record convention from ADR-0006 is unchanged.

## Consequences

- `gh issue list`/`gh issue view` replace reading `TODO.md`; `gh issue create`/`edit`/`close` replace editing it.
- No merge-driver setup step remains in onboarding (`README.md`, `docs/repo-design/development-workspace.md`) or `pyrite-dev verify`.
- Backlog items are now only visible to accounts with repository access (the repository is private), same as before.
- A local clone with no network/`gh` access cannot read or update the backlog; `TODO.md` no longer serves as an offline mirror.
