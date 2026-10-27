---
name: documentation-maintenance
description: Use when changing PyRITE README/docs/public docstrings/generated references/navigation or auditing documentation against implementation.
---

# Documentation Maintenance

Use `docs/repo-design/documentation.md` as the canonical authoring policy.
Classify the reader's *primary need* before editing (Diátaxis):

| Need | Canonical home |
| --- | --- |
| Learn by doing | `docs/notebooks/`; use `tutorial-authoring` and #758 |
| Solve a specific user task | `docs/guides/` (How-to Guides) |
| Look up a public interface | Public docstring, `docs/reference/python-api/` and the transitional `docs/api.md` |
| Understand physical or numerical behavior | `docs/physics/` |
| Inspect independent evidence | `docs/validation/` |
| Read contributor workflow/current architecture | `docs/development/`, `docs/repo-design/` |
| Understand design decisions or proposals | `docs/adr/`, explicitly labeled `docs/research/` |
| Find CLI options, settings, results contracts | Generated CLI help and `docs/reference/` |
| Locate package owners | `docs/repo_map.md`; use `repo-orientation` |
| Track implementation work | GitHub Issues, not public docs |

1. Find the existing canonical owner and current source, tests, and evidence.
2. Shorten, consolidate, or cross-link before creating another page. Tutorials
   teach; how-tos solve; reference specifies; explanation motivates.
3. Add one primary toctree owner with appropriate nested index. When moving a
   page, preserve its published URL via an alias/redirect and update inbound
   links, explicit labels and source references.
4. Preserve public API stability, model assumptions, quantities and units,
   provenance, validation state, and current-versus-proposed distinctions.
5. Keep disposable plans in `agentdocs/`, not public docs.

Example: a Timepix3 acquisition recipe goes in `docs/guides/`; its constructor
signature belongs in generated Python API reference, and response physics in
`docs/physics/`. Do not copy either into the recipe.

## Focused checks

- CLI edits: regenerate and check `pyrite-dev cli-reference` and
  `pyrite-dev cli-deprecations` as applicable.
- Ownership edits: `pyrite-dev repo-map --check`.
- MyST, Sphinx and navigation: `uv run pyrite-dev docs`.
- Skill edits: `uv run pyrite-dev sync-skills`, then
  `uv run pyrite-dev check-skills`.

Inspect the scoped diff and report evidence gaps. For sidebar changes, check
rendered deep navigation and mobile layout as well as link validity.
