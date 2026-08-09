# Repository scripts

Thin standalone wrappers and maintenance utilities for a source checkout.
Reusable Python implementations invoked by `cxr-dev` live in
`cxr_mc.devtools`; wrapper paths remain for documented direct invocations.

- `generate_cli_reference.py`, `generate_cli_deprecations.py`, `smoke.py`, and
  `package_smoke.py` delegate to importable `cxr_mc.devtools` owners.
- `freeze_cli_contract.py` and `refresh_external_cif.py` maintain checked-in
  repository artifacts.
- `cuda_test_profiler.py` and `testing.py` are developer diagnostics, not test
  or validation owners. GPU profiling must run through the remote workflow.
- `hooks/` contains setup hooks; `user_scripts/` contains example operator
  wrappers, not supported `cxr` command implementations.

Prefer the canonical `uv run cxr-dev ...` commands listed in `AGENTS.md`.
