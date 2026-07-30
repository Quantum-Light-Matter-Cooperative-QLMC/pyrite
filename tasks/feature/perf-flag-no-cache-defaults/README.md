# `cxr run -p/--perf` should force `--no-cache` and other required defaults

Branch: `feature/perf-flag-no-cache-defaults`
TODO scope: from `>user<` backlog prose, item 2.

## Problem

`-p`/`--perf` on `cxr run` (`scan.py:304-314`) samples CPU/RAM/GPU/queue/case/
worker/chunk metrics into `performance-profiles/PROFILE/<material>.ndjson`.
If the target material already has a checkpoint, `run_sweep`'s default
`resume=True` (`run.py:320`) skips already-computed cases — so a `--perf` run
against a warm checkpoint samples little or nothing, silently producing a
thin/misleading performance profile instead of erroring or measuring the
requested work.

User asks two related questions:
1. Should `--perf` imply `--no-cache` (force a full recompute so every case is
   actually sampled), rather than requiring the user to remember to pass both?
2. More generally, should `cxr run` force other flag combinations that
   `--perf` (or similar modes) requires, instead of leaving it to the user to
   supply them correctly and failing/misleading silently if they don't?

## Implementation path

Hard-blocked on `feature/run-no-cache` (`tasks/feature/run-no-cache/`) landing
first — `--no-cache` doesn't exist yet on `cxr run`. Once it does:

1. In `scan.py`'s `command()`, when `--perf` is set and `--no-cache` was not
   explicitly passed, force `no_cache=True` before threading it into
   `_run_material` — mirror how `_recompute_options` (`_remote/cli.py:507`)
   or similar existing option-interaction code overrides one flag based on
   another, for the click plumbing pattern already used here.
2. Audit other `--perf`-adjacent options for the same silent-mismatch failure
   mode (e.g. does `--performance-profile` interact with `--quick` or
   `--max-minutes` in a way that produces a misleading sample?) and decide,
   per flag, whether to force the compatible default or raise a `UsageError`
   instead — user's phrasing ("force... rather than request user to supply
   them") suggests a preference for forcing over erroring where the forced
   value is unambiguous, but confirm case by case; forcing silently swallows
   a case where the user *did* want a `--no-cache`-free perf run for some
   reason, so consider a warning line even when forcing.
3. Update `--perf`'s help text (`scan.py:308-313`) to document the forced
   interaction.

## Decisions / open questions

- Confirm the "force, don't error" default holds for every flag interaction
  touched here, or only for `--perf` + `--no-cache` specifically.
- Should forcing print a notice (e.g. "`--perf` implies `--no-cache`") so the
  behavior isn't silently surprising, or is silent forcing acceptable per the
  user's framing?

## Delegation

Small, mechanical once `feature/run-no-cache` lands and the force-vs-error
question is confirmed — `implement-task-lite` tier. Relevant skill:
`cli-ui-ux`.

## Acceptance

- `cxr run <profile> --perf -m <material>` against a material with an
  existing checkpoint samples every case (verify via the emitted
  `.ndjson`'s case count matching the profile's full case count, not just the
  todo-after-resume subset).
- `docs/cli-reference.md` regenerated for `run`'s `--perf` help text.
- `uv run cxr-dev verify` passes.
