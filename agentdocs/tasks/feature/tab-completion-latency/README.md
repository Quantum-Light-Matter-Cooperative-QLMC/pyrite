# Tab completion latency

Branch: `feature/tab-completion-latency`
TODO scope: from `>user<` backlog prose, item 5.

## Problem

Shell tab completion for `cxr` often takes multiple seconds to appear instead
of the near-instant response expected of shell completion. User flags this as
needing a fix without further diagnosis.

## Investigation notes (not yet reproduced/measured)

Local completion providers themselves look cheap: `_material_keys`/
`_profile_keys` (`src/cxr_mc/cli/_completion.py:66-80`) read
`materials.toml` directly via `tomllib` with `@lru_cache(maxsize=1)`, and the
CLI command tree uses `LazyGroup` (`cli/_core.py`) throughout so unrelated
subcommand modules aren't imported just to list names. Two more likely
sources of multi-second latency, neither confirmed yet:

1. **Remote/SSH completion providers block per keystroke.**
   `_completion.py` defines `REMOTE_COMPLETION_TIMEOUT_SECONDS = 1.5` and
   `MAX_REMOTE_CANDIDATES = 100` for job/checkpoint completion that shells out
   over SSH (see `test_job_completion_silences_lookup_failures`,
   `test_remote_checkpoint_completion_includes_variant_stems`,
   `tests/test_cli_completion.py`). If any completion path invoked
   (including non-`remote` subcommands, if the completion callback wiring is
   too broad) triggers one of these SSH round-trips, a 1.5s timeout alone
   accounts for "multiple seconds," worse if the box is slow/unreachable and
   several candidates are probed.
2. **Process/interpreter startup overhead per completion invocation.** Each
   Tab keystroke re-invokes the `cxr` entry point
   (`.venv/bin/cxr` → `cxr_mc.cli:main`) as a fresh Python process. If the
   user's shell rc line or `PATH` resolves `cxr` through a `uv run` wrapper
   rather than the installed venv script directly, every keystroke pays full
   `uv` environment resolution on top of interpreter startup — this project
   has separately documented `uv run` overhead as non-trivial (see
   `CLAUDE.local.md`'s "Canonical commands"/GPU-stack notes). Confirm how the
   user's shell actually invokes `cxr` for completion
   (`type cxr`, contents of the `eval "$(_CXR_COMPLETE=... cxr)"` line
   installed by `cxr completion install`, `cli/completion.py:44`) before
   assuming it's provider logic at fault.

## Implementation path

1. Reproduce: measure wall-clock of a single completion invocation directly,
   e.g. `time COMP_WORDS="cxr run --profile" COMP_CWORD=2 _CXR_COMPLETE=bash_complete cxr`
   (adjust for installed shell/click version's completion protocol), for both
   a local-only completion (e.g. `--profile`) and a remote one (e.g.
   `cxr remote status <TAB>` job-id completion), to isolate which hypothesis
   above (or another) dominates.
2. If SSH-bound: shorten `REMOTE_COMPLETION_TIMEOUT_SECONDS`, cache successful
   lookups briefly, or restrict remote completion to only the subcommands
   that need it (confirm today's wiring doesn't attach it more broadly than
   `cxr remote ...`).
3. If process-startup-bound: confirm `cxr completion install`'s emitted line
   invokes the venv script directly (it should, per `_completion_line`,
   `cli/completion.py:22`) rather than a `uv run`-wrapping shell alias/function
   the user may have layered on top; document the non-`uv run` invocation
   requirement in completion install help/docs if so.

## Delegation

Investigation first — `implement-task` tier once the dominant cause is
confirmed by measurement; the fix itself is likely small once isolated.
Relevant skill: `cli-ui-ux`, `performance`.

## Acceptance

- Measured before/after latency for both a local and a remote completion
  path, with the fix targeting sub-200ms for local-only completions.
- `tests/test_cli_completion.py` covers the fixed behavior (e.g. a timeout
  regression test if the fix is timeout-related).
- `uv run cxr-dev verify` passes.
