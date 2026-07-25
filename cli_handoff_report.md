# CLI migration handoff

Date: 2026-07-25  
Branch: `user-wip`  
Feature implementation commit: `66ddbb4`

## Status

Tasks 1–19 in `cli_implementation_plan.md` are complete.

Current `cxr` entry point uses Click. Local, remote, and line-grid command
families are migrated. Obsolete argparse command-tree wiring is removed. Only
required standalone parsers remain in:

- `src/cxr_mc/line_grid/derive.py`
- `src/cxr_mc/line_grid/job.py`

CLI migration is complete. No remaining task in this handoff.

## Completed work

### P0 correctness and safety

- Central remote host/path/executable and shell-boundary validation.
- Safe line-grid writes with validation before atomic replacement.
- Full line-grid submit flag forwarding and corrected `--brem-step` behavior.
- Correct failure, partial-failure, resumable, and interruption exits.
- Source-checkout-only `regen-golden`.
- Sanitized remote presentation and versioned base64 framing.

### P1 Click migration

- Direct Click 8.4 dependency and shared core in `src/cxr_mc/_cli_core.py`.
- Lazy root group; root help avoids simulation-module imports.
- Shared numeric/beam validation, output helpers, JSON envelopes, legacy
  callback adapter, and exit mapping.
- Migrated every local, remote, and line-grid command.
- Removed obsolete argparse builders and updated parser-specific tests.

Public exit contract:

- success: `0`
- runtime/remote failure: `1`
- usage: `2`
- interruption: `130`
- resumable work: `75`

Human results use stdout. Diagnostics and usage failures use stderr.

### P2 discoverability

- Complete help across all command paths: units, defaults, mutations,
  precedence, incompatibilities, latest-job semantics, and examples.
- Warm root-help median enforced below 200 ms.
- Checked generated reference in `docs/cli-reference.md`.
- Offline material, CSV-material, checkpoint, archive, job-ID, and finite-choice
  completion.
- Remote completion is bounded, prompt-free, failure-silent, and excluded from
  destructive target selection.

### P3 automation

- Read-only JSON for remote jobs/status, line-grid defaults/show, and archives.
- Operation JSON for scan, blaze, rebrem, reline, and remote pull.
- Exactly one versioned envelope on stdout; partial failures retain successful
  results and return nonzero.
- Resumable operation summaries preserve exit 75.

## Compatibility evidence

`tests/data/cli_contract.json` remains immutable post-P0 argparse evidence:

- 42 root/nested help paths
- command and option names
- defaults and dispatch metadata
- mutual exclusions
- nine explicit P0 correction markers

Do not regenerate it with `scripts/freeze_cli_contract.py --write`; that script
captures argparse and predates cutover. `scripts/generate_cli_reference.py`
owns current Click reference generation.

## Verification

- Discoverability/reference gate passed.
- Completion tests: 16 passed; affected CLI suites passed.
- JSON/CLI/reference gate: 223 passed.
- JSON owner compatibility gate: 452 passed.
- Final stale-parser compatibility + JSON/reference gate: 96 passed.
- Lint, typecheck, generated-reference check, and `git diff --check` passed.
- Sandboxed full verification reached `1507 passed, 22 skipped`; only failure
  was sandbox forkserver socket creation (`PermissionError`).
- Unsandboxed full verification reached `1506 passed, 22 skipped`; two known
  failures outside CLI-touched code were confirmed:
  - CuPy-to-NumPy conversion in
    `test_brem_groove_gain_matches_beer_lambert_escape`.
  - `1.191e-7` tolerance miss in
    `test_finite_side_exit_layered_absorption_stays_in_emission_layer`.

Both reproduce the pre-cutover handoff failures and remain out of CLI scope.

## Commands

```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py lint
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py typecheck
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py verify
```

## Commits

- `c16554f` — `fix(cli): harden remote and line-grid workflows`
- `d4bb8fd` — `test(cli): freeze post-P0 argparse contract`
- `ed3fdf1` — `feat(cli): add Click migration core`
- `99637a7` — `feat(cli): migrate command tree to Click`
- `3524883` — `docs(cli): complete help and checked reference`
- `e0ddcb2` — `fix(cli): preserve live help context in reference`
- `5c949b7` — `feat(cli): add safe completion providers`
- `ddbaf6d` — `feat(cli): wire safe shell completion`
- `7629edc` — `feat(cli): add structured JSON adapters`
- `66ddbb4` — `feat(cli): wire structured JSON output`
