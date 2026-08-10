---
name: documentation-maintenance
description: Use when changing README/docs/public docstrings/generated references/repository map, maintaining cross-links, or auditing stale docs/builds in PyRITE.
---

# Documentation Maintenance

Document behavior once at owning artifact:

| Change | Owner |
|---|---|
| Science, install, primary workflow | `README.md` |
| User task or workflow | `docs/guides/` |
| Current physical model or convention | `docs/physics/` |
| Independent evidence or reproduction | `docs/validation/` |
| Exploratory model or unimplemented proposal | `docs/research/` |
| Current implementation/design reference | `docs/repo-design/` |
| Durable architectural decision and rationale | `docs/adr/` |
| CLI contract | Source help + generated `docs/repo-design/cli/cli-reference.md`; use `cli-ui-ux` |
| Public API | Source docstring + `docs/api.md`; use `scientific-library` |
| Ownership/entry point/dependency | `docs/repo_map.md`; use `repo-orientation` |
| Physics claim | Derivation, validation record, ledger |
| Backlog | `TODO.md`; agent work in `agentdocs/`; use `todo-sync` |

Read `docs/repo_map.md` and `docs/repo-design/documentation.md`. Compare prose
with source, tests, and runtime evidence. Add maintained pages to the nearest
section index/toctree; do not use directory placement as a substitute for
published navigation.
Preserve units, assumptions, citations, validation state, and
current-vs-proposed distinction. Prefer relative links and canonical detail.
Keep task plans, handoffs, and reports out of `docs/`; place them under the
owning `agentdocs/tasks/<branch-name>/`, or `agentdocs/plans/` only when they
sequence multiple tasks. Promote durable outcomes to their owning artifact;
use `docs/` only for durable project documentation.

## Verify touched artifacts

- CLI: run `pyrite-dev cli-reference --write` and
  `pyrite-dev cli-deprecations --write`, then their `--check` modes.
- Ownership: `pyrite-dev repo-map`.
- Docs/public API/navigation: `pyrite-dev docs` (clean, offline, warnings as
  errors, generated validation views checked).
- Skills: `pyrite-dev sync-skills`, then `check-skills`.

Inspect scoped diff; report checks and evidence gaps.
