---
name: documentation-maintenance
description: Use when changing PyRITE README/docs/public docstrings/generated references/navigation or auditing documentation against implementation.
---

# Documentation Maintenance

Document behavior once at its durable owner:

| Change | Owner |
|---|---|
| Science, install, primary workflow | `README.md` |
| User task/workflow | `docs/guides/` |
| Current physical model/convention | `docs/physics/` |
| Independent evidence/reproduction | `docs/validation/` |
| Exploratory/unimplemented proposal | `docs/research/` |
| Current implementation/design reference | `docs/repo-design/` |
| Durable architecture decision | `docs/adr/` |
| CLI contract | Source help + generated CLI reference; use `cli-ui-ux` |
| Public API | Source docstring + `docs/api.md`; use `scientific-library` |
| Ownership/entry point/dependency | `docs/repo_map.md`; use `repo-orientation` |
| Backlog/task plan | GitHub Issues |

Read only the relevant owning docs plus source/tests/runtime evidence needed to
verify them. Add maintained pages to the nearest index/toctree. Preserve units,
assumptions, citations, validation state, and current-vs-proposed distinctions.
Prefer relative links and one canonical explanation.

Keep disposable agent plans/reports out of durable docs. Optional `agentdocs/`
working notes are branch-local scratch; promote only durable outcomes.

## Verify touched artifacts

Run only applicable checks:

- CLI: `pyrite-dev cli-reference --write`, `pyrite-dev cli-deprecations --write`,
  then their `--check` modes.
- Ownership/topology: `pyrite-dev repo-map`.
- Docs/public API/navigation: `pyrite-dev docs`.
- Skills: `pyrite-dev sync-skills`, then `check-skills`.

Inspect the scoped diff and report checks/evidence gaps.
