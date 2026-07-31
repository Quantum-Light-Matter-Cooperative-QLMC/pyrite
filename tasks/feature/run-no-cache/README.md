# `--no-cache` for `cxr run` / `cxr remote run`

Branch: `feature/run-no-cache`
TODO scope: new P3 item, "`--no-cache` for run/remote run."

## Problem

`cxr run`/`cxr remote run` always resume: `run_sweep` (`src/cxr_mc/run.py:320`,
`resume=True` default) loads the existing `<material>.pkl` into `results`
before filtering `cases` down to the not-yet-present `todo` set
(`run.py:437-468`). There is no way to force a full recompute of an existing
checkpoint short of manually deleting the pickle. Add `--no-cache` to both
entry points to skip the load and recompute/overwrite every case.

## Key finding — mechanism already does the right thing

`run_sweep(resume=False)` already gives exactly the requested behavior with no
new save-path logic:
- the checkpoint load at `run.py:437` is skipped entirely, so `results` starts
  empty for this material;
- `todo` (`run.py:466`) becomes every case (none are pre-populated);
- `_save()` (`run.py:421-429`) writes `subset = {n: results[n] for n in
  results if ...}` back to `checkpoint_path` — since `results` only holds the
  freshly recomputed cases, the on-disk pickle is overwritten with the new
  data, not merged with the old.

So the local half is a plumbing change, not new logic:
1. `scan.py`: add `@click.option("--no-cache", is_flag=True, help=...)` next to
   `--checkpoint-dir` (`scan.py:290-296`), thread through `command()`'s two
   `invoke_legacy(...)` calls (`scan.py:430`, `459`) as `no_cache=no_cache`.
2. `_run_material` (`scan.py:839`): pass `resume=not getattr(args, "no_cache",
   False)` into the `run_sweep(...)` call (`scan.py:1080`).
3. `run(args)`/`_run_json(args)` need no changes — they only call
   `_run_material`, which already reads `args.no_cache`.

## Open question — chunked remote queue

`cxr remote run` defaults to a chunked SLURM chain (`start_queue`,
`_remote/lifecycle.py:319`; `_chunked_queue_script`, `_remote/scripts.py:272`):
each ~`chunk_minutes` slice invokes `cxr run --max-minutes ...` and *relies on*
loading the previous slice's checkpoint to know what's left to do. If
`--no-cache` is forwarded into every slice's invocation verbatim, slice 2 would
throw away slice 1's freshly-computed results and start over — the chain would
never converge.

`start_rebrem_queue`/`start_reline_queue` sidestep an analogous problem for
their `--redo-all` flag differently (repair-in-place idempotency inside
`repair_brem_wide`, not a load-skip), so that precedent doesn't transfer
directly.

Pick one before wiring the remote side (not resolved here):
- **(a) Restrict to monolithic.** `start_queue` already refuses several flag
  combinations under chunking (`nsys`, `performance_repetitions > 1`,
  `parallel_materials` — `lifecycle.py:367-401`). Add `--no-cache` to that list:
  require `--chunk-minutes 0`. Smallest change, matches existing precedent,
  but silently limits a nominally general flag to one allocation mode.
- **(b) Clear-once at submit time.** `start_queue`/`start_command` delete the
  remote `<material>.pkl` (and manifest) for each target material once, before
  staging the chunked script, and do NOT forward `--no-cache` into the
  per-slice `cxr run` invocation. Every slice then does an ordinary
  resume-from-this-job's-own-progress run. Works for both allocation modes,
  but is a destructive step at submit time (needs a `dry_run`-safe preview and
  should not run inside the box's job script, where a mid-flight failure could
  wipe data with no work yet done to replace it).

Lean (b) for parity with the plain `--chunk-minutes 0` case and because it
doesn't special-case chunking, but confirm before implementing the remote half.

## Checklist

- [ ] Local: `scan.py` `--no-cache` option, threaded through `command()` to
      `run`/`_run_json` via `invoke_legacy`, consumed in `_run_material`.
- [ ] Local: unit test — `_run_material`/`run_sweep` with an existing
      checkpoint and `--no-cache` recomputes and overwrites rather than
      resuming (mirror existing `run_sweep` resume tests in `tests/test_run.py`).
- [ ] Decide (a) vs (b) above for `cxr remote run`; record the choice here.
- [ ] Remote: wire the chosen approach through `start_command`
      (`_remote/cli.py:764`) → `start_queue` (`lifecycle.py:319`) →
      `_queue_script`/`_chunked_queue_script` (`scripts.py:105,272`).
- [ ] `docs/cli-reference.md` regenerated (`uv run cxr-dev docs` per
      `AGENTS.md`'s CLI-change contract) for both `run` and `remote run` help
      text.
- [ ] `CHANGELOG`/help text consistent between local and remote `--no-cache`
      wording.

## Delegation

Small, mechanical once (a)/(b) is decided — `implement-task-lite` tier
(Haiku/Luna). The open question above should be resolved by whoever picks this
up (or flagged back) before the remote-side commit.

## Acceptance

- `cxr run <profile> -m <material> --no-cache` on a material with an existing
  checkpoint recomputes every case and overwrites the pickle (verify via
  `_manifest_save`'s persisted case count / a changed `parameter_sha256`-free
  file mtime, not just "didn't crash").
- `cxr remote run ... --no-cache` behaves per the chosen (a)/(b) design and
  does not silently drop chunk-to-chunk progress.
- `uv run cxr-dev verify` passes.
