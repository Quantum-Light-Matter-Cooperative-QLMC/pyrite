---
name: documentation-maintenance
description: Use when adding, removing, or changing cxr-mc behavior that affects README.md, docs/, public docstrings, generated CLI or API reference, repository map, cross-links, or documentation build health; also use for documentation-only maintenance and stale-doc audits.
---

# Documentation Maintenance

Keep current behavior documented at its owning artifact. Regenerate generated
reference; do not copy same contract into several files.

## Route Changes

Read `docs/repo_map.md`, then inspect changed behavior and existing links.
Update smallest applicable set:

| Change | Owner and required composition |
|---|---|
| Science overview, install, primary workflow | `README.md` |
| Current guide, design rationale, research note | `docs/*.md`; add discoverability in `docs/index.md` and, when authoritative, `docs/README.md` |
| CLI command, option, default, unit, output | Source help text plus generated `docs/cli-reference.md`; use `cli-ui-ux` |
| Public Python API | Source docstring plus `docs/api.md` autosummary coverage; use `scientific-library` |
| Package, entry point, dependency, ownership | `docs/repo_map.md`; use `repo-orientation` |
| Physics equation or claim | Derivation docstring, validation record, and ledger; use `physics-review` or `physics-validation` |
| Tracked backlog state | Branch and main `TODO.md`; use `todo-sync` |

Treat `docs/superpowers/plans/` and `docs/superpowers/specs/` as dated history.
Do not rewrite them to describe later behavior. Add new record only when task
explicitly calls for plan or specification.

## Maintain

1. Compare docs with source, tests, and runtime evidence. Do not preserve stale
   text merely because it is published.
2. Preserve units, assumptions, validation status, citations, and distinction
   between current behavior and proposals.
3. Prefer relative repository links. Update inbound links after moving or
   renaming a page.
4. Write concise task-oriented prose. Link to canonical detail instead of
   duplicating it.
5. Update source and affected documentation in same change.

## Regenerate and Verify

Run checks matching touched artifacts:

- CLI reference:
  `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/generate_cli_reference.py --write docs/cli-reference.md`,
  then repeat with `--check`.
- Repository structure:
  `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py repo-map`;
  reconcile relevant ownership or inventory changes manually.
- Sphinx pages, public docstrings, navigation, or cross-links:
  `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run --group docs sphinx-build -W -b html docs docs/_build/html`.
- Repo-local skills:
  `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py sync-skills`,
  then `check-skills`.

Inspect scoped diff after generation. Report updated artifacts, checks run, and
behavior left undocumented because evidence is unavailable.
