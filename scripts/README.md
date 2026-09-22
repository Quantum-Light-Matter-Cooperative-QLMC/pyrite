# Repository scripts

Thin standalone wrappers and maintenance utilities for a source checkout.
Reusable Python implementations invoked by `pyrite-dev` live in
`pyrite.devtools`; wrapper paths remain for documented direct invocations.

- `generate_cli_reference.py`, `generate_cli_deprecations.py`, `smoke.py`, and
  `package_smoke.py` delegate to importable `pyrite.devtools` owners.
- `freeze_cli_contract.py` and `refresh_external_cif.py` maintain checked-in
  repository artifacts. `release_bremslib_tables.py` builds the BremsLib-derived
  table release from a local BremsLib checkout and pins its index in the package.
- `cuda_test_profiler.py` and `testing.py` are developer diagnostics, not test
  or validation owners. GPU profiling must run through the remote workflow.
- `hooks/` contains setup hooks; `user_scripts/` contains example operator
  wrappers, not supported `pyrite` command implementations.

Prefer the canonical `uv run pyrite-dev ...` commands listed in `AGENTS.md`.
In particular, use `pyrite-dev cli-reference --write|--check`,
`pyrite-dev cli-deprecations --write|--check`, and `pyrite-dev docs`; the
standalone generator files are compatibility wrappers, not documentation
workflow owners.
