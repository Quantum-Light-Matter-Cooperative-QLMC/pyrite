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
  execution and the generated static Read the Docs output. Never maintain a
  second Markdown or application implementation of the same notebook.
- Add an Open in Molab link for that exact source. State units, seeds, backend
  and scientific assumptions visibly; distinguish Monte Carlo sample counts
  from physical populations and flux from acquired detector counts. Do not add
  hidden provenance cells or separate JSON manifests to tutorials.
- Mark quantum/channeling and other experimental science with API admission
  boundaries and physical validation limitations, without implying sign-off.
- HTML Sphinx builds execute each tutorial once on CPU and write disposable
  HTML under `docs/_build/`. Never commit HTML, figures, sessions or provenance
  manifests. Keep notebook sources output-free; builds never rewrite them or
  synchronize release pins. Use a suitably provisioned worker for full builds.
- Cross-link related how-to, API, Physics and Validation pages instead of
  turning the notebook into a manual.

Validate with `uv run marimo check <notebook.py>` and the focused source,
public-import and scientific checks; run
`uv run pyrite-dev docs`. Use `uv run pyrite-dev sync-skills` and
`uv run pyrite-dev check-skills` if this canonical skill changes.
