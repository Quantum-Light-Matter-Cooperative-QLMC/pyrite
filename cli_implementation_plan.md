# CLI implementation plan

Source evidence: `cli_audit_progress.md`. General audit is complete; do not repeat
audit matrices unless changed code invalidates them. Full file is large; only surgical reads of it.

## Handoff — 2026-07-25

Stopping point: P0 tasks 1–7 complete. Task 17 was pulled forward after task 2
and is also complete. P1 tasks 8–9 are complete; resume at task 10, **Migrate
local commands**.

Completed behavior:

- Installed and forward-tested `cli-ui-ux` user-wide for Codex and Claude Code;
  added canonical project skill plus synchronized Claude mirror. `AGENTS.md`
  requires it for every CLI change.
- Centralized validation/rendering now covers remote host, directory, executable,
  SSH/SCP shell words, Bash payloads, and SBATCH directives.
- Line-grid submit forwards geometry/default flags through remote slices; writes
  validate before replacement, reject invalid domains, and roll back catalog and
  provenance together.
- Line-grid defaults/show/derive semantics match decisions below, including real
  `--brem-step` output and stale-golden stderr warning with successful mutation.
- Follow-log failures and interrupts propagate; pull aggregates partial failures;
  public dataset/output/metadata inputs validate.
- `regen-golden` is source-checkout-only with precise installed-wheel failure.
- Remote jobs/status/attach presentation sanitizes controls and uses versioned
  base64 framing instead of `@@`; raw log commands remain raw.
- Post-P0 argparse contract is frozen in `tests/data/cli_contract.json`: 51
  root/nested command nodes with options, defaults, exact help, dispatch, mutual
  exclusions, and explicit P0 correction markers. Every help path plus root
  version/usage stream and exit behavior has focused regression coverage.
- Click 8.4 is now a direct dependency. Shared migration core provides lazy
  command imports, positive/nonnegative numeric and beam-axis parameter types,
  stable result/diagnostic/JSON output, runtime/resumable errors, interruption
  exit 130, and `CliRunner` test helpers.

Verification:

- Combined remote + line-grid integration suite: `308 passed`.
- Full `scripts/dev.py verify`: lint, typecheck, and skill-mirror checks passed.
  Tests reached `1282 passed, 22 skipped`; two failures are outside CLI-touched
  code: CuPy-to-NumPy conversion in
  `test_brem_groove_gain_matches_beer_lambert_escape` and a `1.19e-7` numerical
  tolerance miss in
  `test_finite_side_exit_layered_absorption_stays_in_emission_layer`.
- First sandboxed full run had one forkserver `PermissionError`; escalated rerun
  cleared that infrastructure failure and exposed only the two failures above.

Handoff notes:

- Preserve all current uncommitted P0 changes; no commits were created.
- Preserve `scripts/freeze_cli_contract.py`, `tests/test_cli_contract.py`, and
  `tests/data/cli_contract.json` during local-command migration; adapt snapshot
  plumbing only when Click replaces argparse.

## Decisions

- Execute full queue in phases. Verify P0-P1 before P2-P3.
- Migrate from argparse to Click. Keep `cxr = cxr_mc.cli:main`.
- Install `cli-ui-ux` user-wide for Codex and Claude Code; repository instructions
  must invoke it for every CLI change.
- `line-grid defaults` value flags without `--set`: usage error.
- `line-grid derive --brem-step`: controls derived bremsstrahlung grid.
- Catalog mutation with stale golden: warn and succeed.
- `line-grid regen-golden`: source-checkout-only; installed wheel fails clearly.
- Correct false-success exits and stream routing. Document compatibility changes.

## Required behavior

- Preserve command names, options, defaults, checkpoint semantics, remote safety
  checks, and human output unless this plan names a correction.
- Exit codes: success 0, usage 2, runtime/remote failure 1, interruption 130,
  resumable work 75.
- Diagnostics: stderr. Human results: stdout. Progress must not corrupt piped or
  JSON output.
- Root `--help` and `--version`: lazy imports; warm median target below 200 ms.
- Click migration tests must cover every root, nested help path, dispatch path,
  default, exit, and stream. Use `CliRunner` plus subprocess tests.
- Numeric domains:
  - positive: counts, local durations, energies, lengths, steps, grid sizes;
  - nonnegative: workers, chunk minutes, save cadence, minimum age;
  - preserve zero meanings: serial workers, monolithic remote run, disabled save;
  - enforce emission-angle bounds and reject `beam-uvw=(0,0,0)`.

## Critical defects

- Line-grid submit drops `--tilts`, `--azimuths`, `--thickness`, and
  `--set-default` before generated derive command.
- Line-grid apply validates after replacing live catalog.
- Line-grid set/set-brem/defaults accept invalid domains or silently ignore flags.
- `--brem-step` is accepted but derived JSON always uses 25 eV.
- Follow-log SSH failures and interrupts can exit 0.
- Remote pull masks all per-item failures and can exit 0 with no successful pull.
- Installed `regen-golden` targets absent source-tree `tests/data`.
- Remote environment config crosses SSH/Bash/SBATCH boundaries without central
  validation.

## Security constraints

- `HOST`: host alias only; reject leading dash, controls, whitespace, and shell
  syntax before SSH/SCP argv construction.
- `REMOTE_DIR`: absolute POSIX path; reject controls/newlines; render separately
  for shell words and SBATCH directives.
- `REMOTE_UV`: executable/path, never shell program text.
- Quote every remote shell value. Test every SSH/SCP and generated Bash/SBATCH
  family with hostile values.
- Validate public `pull(dataset=...)`, constrain line-grid remote output names,
  and reject controls in persisted metadata.
- Sanitize status/jobs/attach fields. Raw log commands may remain raw. Replace
  `@@` sentinel framing.

## Automation contract

- `--json`: exactly one UTF-8 object plus newline on stdout; no ANSI, progress,
  trace, warning, or child output.
- Envelope: `schema`, `schema_version: 1`, `ok`, payload, `errors`.
- Any requested-item failure: `ok: false`, nonzero exit, partial results retained.
- IDs are strings; times RFC 3339 UTC; unknown values `null`; collections always
  arrays/objects; units encoded in field names.
- First schemas:
  - `cxr.remote.jobs`: job IDs, command, materials, state, terminal flag, times,
    last event.
  - `cxr.remote.status`: job plus scheduler, progress, and unstructured recent-log
    lines.
  - `cxr.line-grid.defaults`: energy/angle/thickness arrays, brem step, source.
  - `cxr.line-grid.show`: per-material line grids, brem description, provenance.
  - `cxr.archives`: label, path, bytes, record count, readability, error.
  - `cxr.operation-summary`: operation, requested/completed/failed materials,
    checkpoints, elapsed seconds, resumable flag, per-material errors.
- Add NDJSON for follow/attach only when a concrete streaming consumer exists.

## Subagent queue

One task per agent. Run tasks sharing line-grid or remote owners sequentially.

### P0: skill, security, correctness

1. **DONE — Create `cli-ui-ux` skill** — installed user-wide for Codex and Claude Code;
   enforce from repository instructions. Cover naming, help, prompts, streams,
   exits, progress, TTY/Unicode, JSON/NDJSON, completion, accessibility,
   destructive actions, tests, and compatibility. Use `skill-creator`; validate
   and forward-test it.
2. **DONE — Harden remote config** — central validation/rendering for `HOST`,
   `REMOTE_DIR`, and `REMOTE_UV`; hostile SSH/SCP/Bash/SBATCH tests.
3. **DONE — Forward line-grid submit flags** — carry all accepted geometry/default flags
   through CLI, `job.start()`, slice payload, and derive command.
4. **DONE — Make line-grid writes safe** — validate proposed catalog before replacement;
   validate numeric domains; failed writes leave catalog/provenance unchanged.
5. **DONE — Fix line-grid semantics** — reject defaults values without `--set`; fail
   unknown-material show; apply `--brem-step` to output; warn on stale golden.
6. **DONE — Propagate remote failures** — follow-log status and interrupt 130; aggregate
   pull failure; validate dataset/output names/metadata.
7. **DONE — Fix `regen-golden`** — require source checkout with precise installed-wheel
   error; test source and isolated wheel.

### P1: Click migration

8. **DONE — Freeze current contract** — snapshot command tree, options, defaults, help,
   dispatch, exits, and streams. Mark intentional changes from this plan.
9. **DONE — Build Click core** — dependency, lazy `click.Group`, shared decorators,
   parameter types, output/error helpers, and test utilities.
10. **Migrate local commands** — scan, blaze, analyze, check, export, slim,
    rebrem, reline, archive, restore, archives, union, check-config.
11. **Migrate remote commands** — full remote group; preserve previews,
    reservations, liveness checks, and resumable behavior.
12. **Migrate line-grid commands** — full group; preserve P0 corrections and
    required standalone-module entry points.
13. **Cut over** — switch root entry point implementation, remove obsolete
    argparse builders, enforce exit/stream contract, pass full tree tests.

### P2: discoverability

14. **Complete Click help** — summaries, units, defaults, mutation and precedence
    rules, incompatibilities, latest-job semantics, examples, startup benchmark.
15. **Generate CLI reference** — checked command tree, docs links, cluster
    defaults, repo-map inventory, historical-doc labels, single-source version.
16. **Add Click completion** — materials, comma-separated materials, checkpoints,
    archive labels, job IDs, and choices. Network lookup bounded, silent, and
    empty on failure. Never complete destructive clear/reap paths.

### P3: presentation and automation

17. **DONE — Sanitize remote presentation** — status/jobs/attach controls and framing.
18. **Add read-only JSON** — remote jobs/status, line-grid defaults/show, archives.
19. **Add operation JSON** — scan/blaze/rebrem/reline and remote pull.

## Dispatch

- Task 1 gates every CLI-changing task.
- After task 1: run 2, 3, and 7 in parallel.
- Then: 3 → 4 → 5; 2 → 6 and 17.
- Migration: 8 → 9 → parallel 10/11/12 → 13.
- Finish: 13 → 14/16/18; 14 → 15; 18 → 19.
- Each agent reports changed files and exact focused checks. Integrator reviews
  scoped diffs and runs:

```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py verify
```
