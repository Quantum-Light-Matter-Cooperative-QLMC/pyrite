# Repository scripts

Thin standalone wrappers and maintenance utilities for a source checkout.
Reusable Python implementations invoked by `pyrite-dev` live in
`pyrite.devtools`; wrapper paths remain for documented direct invocations.

- `generate_cli_reference.py`, `generate_cli_deprecations.py`, `smoke.py`, and
  `package_smoke.py` delegate to importable `pyrite.devtools` owners.
- `freeze_cli_contract.py`, `refresh_external_cif.py`, and
  `export_docs_demo.py` (browser demo data and README preview) maintain
  checked-in repository artifacts. `release_bremslib_tables.py` builds the BremsLib-derived
  table release from a local BremsLib checkout and pins its index in the package.
- `docs_figures.py` regenerates the dependency-free documentation SVG
  schematics in `docs/_static/figures/`; `pyrite_logo.py` generates the
  physics-derived SVG/PNG wordmark.
- `release_elsepa_tables.py`, `release_sbethe_tables.py`,
  `release_bremslib_tables.py`, `release_epdl_table.py`,
  `release_brem_lattice_share_table.py` (the universal lattice-coherent
  bremsstrahlung share kernel, #1250) and `release_epdl_differential.sh` are
  maintainer-only hosted-table release builders (see `docs/repo-design/data-distribution-and-repository-size.md`).
- `__init__.py` makes the directory importable for wrapper tests.
- `hooks/sync-skills.sh` is the pre-commit hook that regenerates
  `.claude/skills` from `.agents/skills`.
- One-off benchmarks and evidence scripts belong in `checks/`, not here.

Prefer the canonical `pyrite-dev ...` commands listed in `AGENTS.md`.
In particular, use `pyrite-dev cli-reference --write|--check`,
`pyrite-dev cli-deprecations --write|--check`, and `pyrite-dev docs`; the
standalone generator files are compatibility wrappers, not documentation
workflow owners.
