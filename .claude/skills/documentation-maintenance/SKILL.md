---
name: documentation-maintenance
description: Use when changing README/docs/public docstrings/generated references/repository map, maintaining cross-links, or auditing stale docs/builds in PyRITE.
---

# Documentation Maintenance

Document behavior once at owning artifact:

| Change | Owner |
|---|---|
| Science, install, primary workflow | `README.md` |
| Guide, rationale, research note | `docs/*.md`; link from docs navigation |
| CLI contract | Source help + generated `docs/cli-reference.md`; use `cli-ui-ux` |
| Public API | Source docstring + `docs/api.md`; use `scientific-library` |
| Ownership/entry point/dependency | `docs/repo_map.md`; use `repo-orientation` |
| Physics claim | Derivation, validation record, ledger |
| Backlog | `TODO.md`; agent work in `agentdocs/`; use `todo-sync` |

Read `docs/repo_map.md`. Compare prose with source, tests, and runtime evidence.
Preserve units, assumptions, citations, validation state, and
current-vs-proposed distinction. Prefer relative links and canonical detail.
Keep task plans, handoffs, and reports out of `docs/`; place them under the
owning `agentdocs/tasks/<branch-name>/`, or `agentdocs/plans/` only when they
sequence multiple tasks. Promote durable outcomes to their owning artifact;
use `docs/` only for durable project documentation.

## Verify touched artifacts

- CLI: run generator `--write`, then `--check`.
- Ownership: `pyrite-dev repo-map`.
- Docs/public API/navigation: strict `sphinx-build -W`.
- Skills: `pyrite-dev sync-skills`, then `check-skills`.

Inspect scoped diff; report checks and evidence gaps.
