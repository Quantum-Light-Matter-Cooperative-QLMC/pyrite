---
name: tutorial-authoring
description: Use when creating or maintaining public-facing Marimo/MoLab scientific tutorials in docs/notebooks, including static exports and reproducibility.
---

# Tutorial Authoring

Follow `docs/repo-design/documentation.md` and the curriculum in PyRITE issue
#758. Tutorials **teach by doing** and differ from task-oriented how-tos and
packaged marimo applications (which use `notebook-workflow`).

- Keep each notebook independently executable, with one clear objective,
  short setup, several public-API cells, one or two useful figures, a concise
  physical interpretation, and optional variations. Aim for 6–10 code cells.
- Use canonical PyRITE imports, explicit units, reproducible seeds, and honest
  performance limits. Never import from `pyrite.apps` or private modules.
- The canonical `docs/notebooks/*.py` Marimo source must drive both local/MoLab
  execution and the checked static Read the Docs output. Never maintain a
  second Markdown or application implementation of the same notebook.
- Add an Open in Molab link for that exact source. Record execution provenance,
  backend and relevant model/table identity; distinguish Monte Carlo sample
  counts from physical populations and flux from acquired detector counts.
- Mark quantum/channeling and other experimental science with API admission
  boundaries and physical validation limitations, without implying sign-off.
- Ensure a standard Sphinx build consumes precomputed outputs only; CUDA and
  expensive Monte Carlo regenerate explicitly, not in CI docs rendering.
- Cross-link related how-to, API, Physics and Validation pages instead of
  turning the notebook into a manual.

Validate with `uv run marimo check <notebook.py>` and #758's source/output
freshness and public-import checks when that infrastructure lands; run
`uv run pyrite-dev docs`. Use `uv run pyrite-dev sync-skills` and
`uv run pyrite-dev check-skills` if this canonical skill changes.
