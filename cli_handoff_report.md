# CLI migration handoff

Date: 2026-07-25  
Branch: `user-wip`  
Current implementation commit: `99637a7`

## Status

Tasks 1–13 and 17 in `cli_implementation_plan.md` are complete.

Current `cxr` entry point uses Click. Local, remote, and line-grid command
families are migrated. Obsolete argparse command-tree wiring is removed. Only
required standalone parsers remain in:

- `src/cxr_mc/line_grid/derive.py`
- `src/cxr_mc/line_grid/job.py`

Resume with tasks 14, 16, and 18 in parallel.

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
- Lazy root group in `src/cxr_mc/cli.py`.
- Root `--help` renders command summaries without importing simulation modules.
- Shared finite/positive/nonnegative numeric types, beam-axis validation,
  output helpers, JSON envelope helper, legacy callback adapter, and exit mapping.
- Migrated all local commands.
- Migrated all 14 remote subcommands.
- Migrated all 12 line-grid subcommands.
- Removed obsolete argparse builders and converted parser-specific tests to
  `CliRunner`.

Public exit contract:

- success: `0`
- runtime/remote failure: `1`
- usage: `2`
- interruption: `130`
- resumable work: `75`

Human results use stdout. Diagnostics and usage failures use stderr.

## Compatibility evidence

`tests/data/cli_contract.json` is immutable post-P0 argparse baseline:

- 42 root/nested help paths
- command and option names
- defaults and dispatch metadata
- mutual exclusions
- nine explicit P0 correction markers

`tests/test_cli_contract.py` checks Click tree against baseline command and option
names and executes every frozen help path.

Do not regenerate baseline with current
`scripts/freeze_cli_contract.py --write`. Script captures argparse and predates
cutover. Task 15 should either adapt it into Click reference generation or
retire it while preserving JSON baseline.

## Verification completed

- Local migration focused suite: 150 passed.
- Remote migration focused suite: 266 passed.
- Line-grid migration focused suite: 142 passed.
- Independent boundary review fixes: 343 passed.
- Final remote/line-grid argparse cleanup: 311 passed.
- Combined P1 CLI gate passed across root/core/contract/local/remote/line-grid
  and affected legacy suites.
- Full lint passed.
- Full typecheck passed.
- `git diff --check` passed.

Full repository verification has not been rerun after Click cutover.

Last pre-cutover full verification reached `1282 passed, 22 skipped` with two
known failures outside CLI-touched code:

- CuPy-to-NumPy conversion in
  `test_brem_groove_gain_matches_beer_lambert_escape`.
- `1.19e-7` tolerance miss in
  `test_finite_side_exit_layered_absorption_stays_in_emission_layer`.

Treat those as pre-existing until current full verification confirms otherwise.

## Remaining tasks

### Parallel after task 13

14. Complete Click help:
   summaries, units, defaults, side effects, precedence, incompatibilities,
   examples, and root startup benchmark below 200 ms warm median.

16. Add Click completion:
   materials, comma-separated materials, checkpoints, archive labels, job IDs,
   and choices. Keep dynamic completion bounded, silent, and side-effect free.

18. Add read-only JSON:
   remote jobs/status, line-grid defaults/show, and archives. Emit exactly one
   versioned envelope on stdout with no progress, warnings, prompts, or ANSI.

### Dependencies

15. Generate checked CLI reference after task 14.

19. Add operation JSON after task 18:
   scan, blaze, rebrem, reline, and remote pull.

Then run full `scripts/dev.py verify`.

## Suggested ownership

- Agent A: task 14, then task 15.
- Agent B: task 16.
- Agent C: task 18, then task 19.
- Integrator: compatibility review, startup benchmark, combined tests, full
  verification, plan update, and commits.

Keep command-family ownership disjoint. Shared edits to `_cli_core.py`,
`cli.py`, generated reference files, and `cli_implementation_plan.md` belong to
integrator unless coordinated explicitly.

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
