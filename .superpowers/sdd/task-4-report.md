# Task 4 Report: chained submission in `remote.py`

## Status: DONE

Commit: `b76d584` — "feat(remote): chunked self-resubmitting SLURM chain (default 10 min slices)"
(branch `worktree-remote-chunked-jobs`, on top of Task 3 `93ee301`).

## What was implemented

`src/cxr_mc/remote.py`:

- `import math` added.
- `_chunked_queue_script(jobid, materials, quick, workers, chunk_minutes)` — new payload
  generator, copied from the brief's sketch essentially verbatim (grep-resume loop over
  `completed:`/`failed:` markers, per-material remaining-budget computation via
  `scan.py --max-minutes`, three-way exit handling (0 / 75 / hard-fail), unresolved/failures
  accounting, terminal `done`/`done with N warning(s)` states, `STOP` sentinel short-circuit,
  state-first handoff (`queued slice <k+1>` written *before* `sbatch`), fail-closed
  numeric-SID-validated resubmission with `--nice=10000`, and `slurm_job_id:` appended to
  `meta` only after a valid SID).
- `_slurm_batch_script(..., time_limit: str = SLURM_TIME)` — new keyword-only parameter;
  `#SBATCH --time={SLURM_TIME}` replaced with `{time_limit}`. The `finish` trap is now
  terminal-only: `"queued slice"*) ;;` (empty body — no release, since a next slice is
  already pending) is checked first, then `done*|FAILED*|cancelled*|cancelling*) release_reservations ;;`,
  then the wildcard `*)` writes `FAILED (exit $status) ...` and also releases. Previously
  `release_reservations` ran unconditionally after the `case`, on every exit — that was the
  bug the handoff needed fixed.
- `_submit_slurm_command(jobid, reservation_stems=None, *, nice: bool = False)` — emits
  `sbatch --parsable --nice=10000` when `nice`, else the original `sbatch --parsable`.
- `start_queue(materials, quick=False, workers=None, no_sync=False, dry_run=False, parallel_materials=None, chunk_minutes=10.0)`
  — `chunked = chunk_minutes > 0`. Raises `SystemExit` mentioning `chunk-minutes 0` if chunked
  and `parallel_materials` is explicitly given. In the chunked branch: `parallel_materials`
  is forced to `None`, payload is `_chunked_queue_script(...)`, and
  `time_limit = str(max(1, math.ceil(chunk_minutes * 3)))`. In the monolithic branch:
  `parallel_materials` is resolved (default `DEFAULT_PARALLEL_MATERIALS=2` when unset) via
  the existing `_validate_parallel_materials`, payload is the original `_queue_script(...)`,
  `time_limit = SLURM_TIME`. `_submit_slurm_command(jobid, stems, nice=chunked)` — every
  chunked submission, including slice 0, carries `--nice=10000`. The final print statement
  now shows `chunk minutes: N (self-resubmitting)` for chunked jobs instead of
  `parallel materials: N`.
- `_queue_metadata(..., parallel_materials: int | None = DEFAULT_PARALLEL_MATERIALS, chunk_minutes: float = 0)`
  — new `chunk_minutes: <value>` metadata line (right after `parallel_materials:`). Both
  params gained explicit type annotations because pyright could not otherwise verify the
  union types flowing from `start_queue`'s two branches.
- `remote_scan(material, quick=False, workers=None)` — the pre-existing compatibility helper
  now passes `chunk_minutes=0` explicitly to `start_queue`, so it keeps its original
  monolithic behavior now that chunked is `start_queue`'s new default. (Required per the
  brief's own framing that `remote_scan()` "stay[s] monolithic ... apart from the shared trap
  change" — without this it would silently become chunked.)
- `start_zhai_queue` — untouched (still relies on `_slurm_batch_script`'s and
  `_submit_slurm_command`'s defaults: `time_limit=SLURM_TIME`, `nice=False`).
- `_cli_scan`: `parallel_materials=getattr(args, "parallel_materials", None)`,
  `chunk_minutes=getattr(args, "chunk_minutes", 10.0)` — `getattr` with fallbacks (matching
  the file's existing style for this function) so hand-built `argparse.Namespace` objects in
  older tests that predate `chunk_minutes` still work without every call site being updated.
- `_cli_start`: `chunk_minutes=args.chunk_minutes` (direct attribute access, matching the
  existing `parallel_materials=args.parallel_materials` style — `_cli_start` is only invoked
  through the real parser in tests, so no defensive `getattr` needed).
- Both `scan` and `start` argparse subparsers: `--parallel-materials` default changed from
  `DEFAULT_PARALLEL_MATERIALS` to `None` (choices unchanged); new `--chunk-minutes`
  (`type=float, default=10.0`) with help text noting `0` = monolithic.

## TDD evidence

**RED** — `uv run python scripts/dev.py test tests/test_remote.py -k "chunk or parallel_materials"`
(before implementation, after writing the 4 new Step-1 tests):
```
FAILED tests/test_remote.py::test_chunked_dry_run_emits_chain_script - assert...
FAILED tests/test_remote.py::test_chunk_minutes_zero_emits_monolithic_script
FAILED tests/test_remote.py::test_parallel_materials_rejected_in_chunked_mode
FAILED tests/test_remote.py::test_cli_start_chunk_flags - SystemExit: 2
4 failed, 113 deselected in 1.30s
```
All four failed for the expected reason — no `chunk_minutes`/`--chunk-minutes` parameter
existed yet (argparse `unrecognized arguments: --chunk-minutes 0`, and assertion mismatches
against monolithic-only script content once `start_queue` rejected/ignored the new kwarg).

**GREEN** — same command after implementation:
```
..........
10 passed, 107 deselected in 1.62s
```

**Full file** — `uv run python scripts/dev.py test tests/test_remote.py`:
```
........................................................................ [ 61%]
.............................................                            [100%]
117 passed in 2.10s
```

**Lint** — `uv run python -m ruff check src/cxr_mc/remote.py tests/test_remote.py`: `All checks passed!`

**Typecheck** — `uv run python scripts/dev.py typecheck`: `0 errors, 0 warnings, 0 informations`
(required adding explicit `int | None` / `float` annotations to `_queue_metadata`'s
`parallel_materials`/`chunk_minutes` params, and restructuring `start_queue` so the
`parallel_materials` reassignment happens inside the same `if chunked: ... else: ...` block
that consumes it — pyright could not otherwise correlate the earlier `if not chunked:` guard
with the later branch across the merge point).

## Existing tests updated

Per the brief's "update existing dry-run tests that assert monolithic-only content" guidance:

- `test_start_writes_static_metadata_before_sbatch` — added `chunk_minutes=0` so its explicit
  `parallel_materials=3` stays legal.
- `test_remote_start_defaults_to_two_parallel_materials` → renamed
  `test_remote_start_defers_parallel_materials_default_to_start_queue`. Since the CLI no
  longer bakes in a default (parser default is now `None`), this test's old assertion
  (`parallel_materials == 2`) no longer reflects real CLI behavior when `start_queue` is
  mocked out — the resolution to `2` now happens inside `start_queue` itself, which the new
  `test_chunk_minutes_zero_emits_monolithic_script` exercises directly
  (`"parallel_materials=2" in out`). The renamed test now asserts the CLI forwards `None`
  through (`--chunk-minutes 0` given, `--parallel-materials` omitted).
- `test_remote_start_accepts_parallel_materials_three_and_four`,
  `test_remote_scan_forwards_parallel_materials` — added `--chunk-minutes 0` so the explicit
  `--parallel-materials` values they assert stay a legal (not merely un-validated-because-mocked)
  combination.
- `test_start_queue_rejects_parallel_materials_above_four` — added `chunk_minutes=0` so the
  call reaches `_validate_parallel_materials`'s `"between 1 and 4"` message instead of the new
  chunked-mode rejection.
- `test_remote_start_rejects_parallel_materials_above_four` — left unchanged: the
  `choices=range(...)` argparse validation for `5` fires before any chunked/monolithic
  distinction, so it's mode-independent.
- `test_remote_dry_run_describes_sbatch_submission`,
  `test_slurm_batch_script_requests_the_lab_gpu_profile`,
  `test_submit_command_uses_sbatch_parsable_and_records_scheduler_id`, and the various
  non-dry-run `start_queue`/`_submit_slurm_command`/reservation tests that don't touch
  `parallel_materials` — left unchanged as mode-independent (chunked being the new default for
  `start_queue` doesn't break their assertions, since `"sbatch --parsable"` is a substring of
  `"sbatch --parsable --nice=10000"`).

## New tests added (brief Step 1, verbatim structure)

- `test_chunked_dry_run_emits_chain_script` — asserts `--max-minutes`,
  `sbatch --parsable --nice=10000`, `queued slice`, `FAILED (slice resubmission)`, the STOP
  sentinel check, `failed: $m`, `#SBATCH --time=30` (3×10min backstop), the trap's
  `"queued slice"*` case, and specifically that the case body is empty
  (`'"queued slice"*) ;;' in out`) — proving the handoff case releases nothing.
- `test_chunk_minutes_zero_emits_monolithic_script` — asserts `#SBATCH --time=UNLIMITED`,
  `parallel_materials=2` (verifying the monolithic default resolution inside `start_queue`),
  and the absence of `--max-minutes`.
- `test_parallel_materials_rejected_in_chunked_mode` — asserts `SystemExit` matching
  `"chunk-minutes 0"`.
- `test_cli_start_chunk_flags` — exercises both the legal (`--chunk-minutes 0
  --parallel-materials 3`) and illegal (`--parallel-materials 3` with the chunked default)
  combinations through `remote.main`.

## Manual end-to-end verification (self-review)

Beyond the unit tests, I generated the chunked payload and the full SLURM wrapper via
`remote._chunked_queue_script(...)` / `remote._slurm_batch_script(...)` and ran them under a
real `bash` with fake `uv`/`sbatch` stand-ins in a scratch directory, to catch quoting bugs
the f-string escaping (`{{`/`}}`, `\\"`, line-continuation backslashes) could introduce and
that string-matching tests wouldn't reveal:

- `bash -n` syntax-checked both the raw chunked payload and the full wrapped scripts
  (chunked and monolithic) — no syntax errors.
- **Resume**: slice 1 with `hopg` succeeding and `hbn` returning 75 (budget) correctly wrote
  `completed: hopg` / left `hbn` unmarked, computed `unresolved=1`, wrote
  `queued slice 1 ...` *before* calling the fake `sbatch` (captured via a side log), appended
  `slurm_job_id: 99999` to `meta`. Re-running the same `run.sh` (simulating slice 2) correctly
  skipped `hopg` via the `grep -qx "completed: hopg"` guard, ran `hbn` to completion, and
  wrote the terminal `done [2/2] ...` state with no further resubmission.
- **Hard failure**: a material exiting 3 (not 0, not 75) got `failed: <m>` appended to `log`
  and the `warning at ...` state line, and reaching `unresolved=0` (both resolved: one
  completed, one failed) produced a terminal `done with 1 warning(s) ...` state with no
  resubmission — matching "never retried" semantics.
- **STOP sentinel**: with `$JOBDIR/STOP` present and a material still unresolved, the script
  exited 0 without writing a `queued slice` state and without calling `sbatch`. Run through
  the full wrapper (with real reservation directories seeded), the `finish` trap's wildcard
  branch caught the stale non-terminal state and wrote `FAILED (exit 0) ...`, correctly
  releasing the reservations — i.e., a user-requested stop still ends in a terminal,
  reservation-clean state.
- **Fail-closed resubmission**: a fake `sbatch` returning nonzero produced
  `FAILED (slice resubmission) ...` with no `slurm_job_id` appended to `meta`, and exit 1
  propagated out of `run.sh`.
- **Numeric-SID validation**: a fake `sbatch` exiting 0 but printing a non-numeric string
  (`not-a-number`) was correctly rejected by the `case "$SID" in ''|*[!0-9]*)` guard —
  `FAILED (slice resubmission) ...`, no `meta` append, exit 1.

## Files changed

- `src/cxr_mc/remote.py`
- `tests/test_remote.py`

## Concerns / notes for Task 5 and reviewers

- The STOP-sentinel path intentionally leaves the job's `state` file untouched at the point
  of the `[ -f "$JOBDIR/STOP" ] && exit 0` check (matching the brief's sketch exactly). This
  means a user-stopped chain surfaces as `FAILED (exit 0) ...` via the wrapper's `finish` trap
  wildcard branch, not as a distinct "stopped"/"cancelled" state. This is almost certainly
  intentional (it's the simplest way to guarantee reservations get released on a manual stop
  without adding a new terminal-state vocabulary word), but it does mean Task 5's poller
  cannot distinguish "chain hit a real script crash" from "operator dropped a STOP file" by
  state string alone — both read `FAILED (exit 0) <timestamp>`. Flagging this now in case
  Task 5 or a human wants a more specific state for the STOP path.
- `_queue_metadata`'s `parallel_materials` field is recorded as Python's `None` (i.e., the
  literal text `parallel_materials: None`) for chunked jobs, since chunked mode has no notion
  of in-allocation parallelism. Not covered by any assertion; flagging in case a later task
  wants a cleaner sentinel there.
- While iterating I ran `uv run python scripts/dev.py format`, which invokes `ruff format .`
  repo-wide and reformatted numerous unrelated files (including `src/cxr_mc/sweep.py`, which
  I was explicitly told never to touch). I reverted every file outside
  `src/cxr_mc/remote.py` / `tests/test_remote.py` with `git checkout --`, confirmed
  `sweep.py`/`mats_to_sim.toml`/`test_sweep.py` are untouched in the final diff and commit,
  and used `ruff format --diff` scoped to just the two files I changed for the final
  formatting check instead of the repo-wide command. Worth noting for future tasks in this
  worktree: `scripts/dev.py format` is not safe to run blindly when the tree has other
  pre-existing dirty files. (There is one pre-existing, out-of-scope dirty file,
  `.superpowers/sdd/task-3-report.md`, that predates this session and was left untouched
  throughout; it is not part of this commit.) This report file itself
  (`.superpowers/sdd/task-4-report.md`) was found pre-populated with an unrelated report from
  a different plan's "Task 4" (finite spectrum escape-distance) and has been overwritten with
  this task's report.

## Review-fix follow-up (commit `8a69b8a`)

Verdict addressed: one Critical, two Important, two Minors (Minor 6 skipped per instruction).

### Critical 1: `--nice=10000` dead on the live path — FIXED

`start_queue` built the nice-aware `submit` string but the real (non-dry-run) path went
through `_submit_staged_job(jobid, stems)`, which rebuilt the command via
`_submit_slurm_command(jobid, stems)` with no `nice` — so live chunked slice-0 submissions
never carried the flag. Fix: `_submit_staged_job(jobid, stems, *, nice: bool = False)`
forwards `nice` to `_submit_slurm_command`, and `start_queue` calls
`_submit_staged_job(jobid, stems, nice=chunked)`. The only other caller,
`start_zhai_queue` (line ~1155), was checked and left on the default `nice=False`.

### Important 3: real-path coverage — ADDED

`test_real_chunked_submission_carries_the_nice_flag` mirrors
`test_start_writes_static_metadata_before_sbatch`'s harness (mocked `_ssh_capture`
capturing every remote command), runs default-chunked `start_queue(["hopg"], no_sync=True)`,
and asserts every CAPTURED command containing `sbatch` includes
`sbatch --parsable --nice=10000`. The monolithic counterpart assertion
(`"--nice=10000" not in submissions[1][-1]`) was added to
`test_start_writes_static_metadata_before_sbatch`.

### Important 2: STOP-sentinel mislabeling — FIXED (controller-approved deviation from the brief)

`_chunked_queue_script`'s STOP branch is now:
`[ -f "$JOBDIR/STOP" ] && { echo "cancelled (stop requested) $(date -Is)" > "$JOBDIR/state"; exit 0; }`
The `cancelled` prefix hits the finish trap's terminal branch, so reservations are released
and Task 5's poller reads it as terminal (this resolves the first concern flagged in the
original report). Dry-run assertion `"cancelled (stop requested)" in out` added to
`test_chunked_dry_run_emits_chain_script`.

### Minor 4: `chunk_minutes:` metadata assertions — ADDED

`test_start_writes_static_metadata_before_sbatch` now asserts `chunk_minutes: 0`
(monolithic); the new real-path test asserts `chunk_minutes: 10.0` (chunked default).

### Minor 7: help text — FIXED

Both parsers' `--parallel-materials` help now reads "... only with --chunk-minutes 0,
which defaults it to 2 (max: 4)" — making clear the default of 2 applies in monolithic
mode and is resolved inside `start_queue`, not by argparse.

### TDD evidence for the fixes

**RED** — with the test changes in place and the source fixes stashed
(`git stash push -- src/cxr_mc/remote.py`, i.e. against commit `b76d584`):
```
uv run python scripts/dev.py test tests/test_remote.py -k "chunked_dry_run or real_chunked or static_metadata"
FAILED tests/test_remote.py::test_chunked_dry_run_emits_chain_script
  E  assert 'cancelled (stop requested)' in '# job ...'
FAILED tests/test_remote.py::test_real_chunked_submission_carries_the_nice_flag
  E  assert all("sbatch --parsable --nice=10000" in command ...)  ->  False
2 failed, 1 passed, 115 deselected in 1.22s
```
The real-path test fails against the old source with exactly the Critical-1 bug (live
sbatch commands lack the flag); the dry-run test fails on the missing cancelled state.
(`test_start_writes_static_metadata_before_sbatch` passes under the stash because the
`chunk_minutes:` metadata line and monolithic no-nice behavior already existed in
`b76d584` — those assertions are pure coverage additions.)

**GREEN** — after `git stash pop`:
```
uv run python scripts/dev.py test tests/test_remote.py
118 passed in 1.70s

uv run python scripts/dev.py lint
All checks passed!

uv run python scripts/dev.py typecheck
0 errors, 0 warnings, 0 informations
```
(Canonical `scripts/dev.py lint` used this round, per review note.)

### Manual re-verification of the STOP path

Re-ran the full-wrapper scratch harness (fake `uv`/`sbatch`, real reservation dirs seeded,
`STOP` present, `hbn` unresolved): `bash -n` clean; script exited 0 **without** invoking
`sbatch` (the fake would have failed loudly); state file read
`cancelled (stop requested) <timestamp>` (previously `FAILED (exit 0) ...`); both
reservation directories were released by the trap's terminal branch.

### Remaining concerns

None new. The original report's STOP-state concern is resolved by Important 2. The
`parallel_materials: None` metadata sentinel (Minor 6) remains as-is per instruction.
