# Task 5 report: stop sentinel + chain-aware attach with watchdog

(Overwrites a stale `task-5-report.md` left over from an unrelated earlier
plan — "finite footprint sweep"; the brief directs this task's report here.)

## Commit

- `b4b77d6` — `feat(remote): STOP sentinel and chain-aware attach with broken-chain watchdog`
- Files staged/committed: `src/cxr_mc/remote.py`, `tests/test_remote.py` only.

## What was implemented

`src/cxr_mc/remote.py`:

- `_POLL_GRACE_POLLS = 15` module constant (~30 s at the 2 s poll), with a comment
  explaining the inter-slice-latency rationale.
- `_is_terminal_state(state)`: `state.startswith(("done", "FAILED", "cancelled"))`.
- `_poll_chain(jobid) -> (state, slurm_live, records)`: ONE ssh round-trip emitting
  `@@STATE` / `@@SID` / `@@SQUEUE` / `@@PROGRESS` sections. Re-reads the latest
  recorded `slurm_job_id` from meta each poll (chains hop SIDs at slice
  boundaries), guards the squeue query behind a numeric-SID `case`, reuses
  `_squeue_state_command(..., retired="STATE=")`, and parses progress snapshots
  with `_parse_progress_records`.
- `_attach_progress_dashboard`: bar setup and the metadata guard (materials + a
  recorded SID) unchanged. The poll loop now uses `_poll_chain`; breaks cleanly
  on a terminal state and prints the LAST POLLED state (no extra `_job_state`
  round-trip, per the brief's parenthetical); a missed-counter resets when a
  SLURM job is live and increments when not; at `_POLL_GRACE_POLLS` misses it
  breaks as broken, prints "chain appears broken ... check
  `cxr remote status <jobid>`", and returns False. Ctrl-C behavior
  (viewer-only disconnect) unchanged.
- `_attach_log_stream`: the old wait-for-first-SID loop (30 x sleep 1) and the
  single-SID squeue loop are replaced by one shell-side loop with the same
  watchdog: terminal-prefix check on `$D/state` (`done*|FAILED*|cancelled*`),
  per-poll re-read of the latest SID, `LIVE` from `_squeue_state_command`,
  missed-counter with the threshold interpolated from `_POLL_GRACE_POLLS` at
  `sleep 2`, broken-chain message to stderr. Tail startup/teardown and the
  final `--- job finished ---` print unchanged.
- `_stop_jobid`: the remote command now writes the STOP sentinel
  (`: > "$D/STOP"; `) BEFORE `scancel` (spec 3b), with a comment on why the
  ordering matters (a slice already past its scan loop still sees STOP and
  terminates instead of re-queueing).
- Removed `_read_progress_records`: its only caller was the old dashboard loop
  (its remote command body now lives inside `_poll_chain`'s `@@PROGRESS`
  section). Verified nothing else in src/ or tests/ references it.

`tests/test_remote.py`:

- Appended the brief's four tests verbatim (modulo formatter line-wrapping):
  `test_poll_chain_parses_sections`,
  `test_attach_dashboard_survives_slice_gap_and_ends_terminal`,
  `test_attach_dashboard_watchdog_exits_on_broken_chain`,
  `test_stop_writes_stop_sentinel_before_scancel`, plus the `_poll_payload`
  helper.
- Updated three pre-existing dashboard tests that mocked the old
  `_read_progress_records` / `_slurm_state` / `_job_state` loop to mock
  `_poll_chain` instead:
  - `test_attach_dashboard_allocates_every_material_row_without_dynamic_width`:
    single terminal poll `("done [3/3] now", False, {})`.
  - `test_dashboard_attach_closes_bars_and_prints_final_state`: two-poll
    sequence (running + live, then `done with 1 warning(s) ...` terminal);
    still asserts bar totals, closure, and that the final print carries the
    warning state — now sourced from the last polled state.
  - `test_dashboard_attach_ctrl_c_disconnects_viewer_only`: `_poll_chain`
    raises KeyboardInterrupt.
- No changes needed to the log-stream attach tests
  (`test_attach_uses_stdin_closed_ssh_for_live_view`,
  `test_attach_retries_until_a_queued_job_creates_its_log`,
  `test_attach_returns_false_when_the_viewer_is_interrupted`) or to the stop
  tests (`test_stop_waits_for_scheduler_cancellation_before_releasing_reservations`,
  `test_stop_retains_reservations_if_squeue_cancellation_query_fails`,
  `test_stop_jobid_uses_scancel_not_kill`) — their assertions hold against the
  new commands and were confirmed passing.

## TDD evidence

- Step 2 (red): `uv run python scripts/dev.py test tests/test_remote.py -k
  "poll_chain or stop_sentinel"` → `2 failed, 120 deselected in 1.77s`
  (`test_poll_chain_parses_sections`: AttributeError, `_poll_chain` missing;
  `test_stop_writes_stop_sentinel_before_scancel`: ValueError, "STOP"
  substring not found).
- The two new dashboard tests could not be run pre-implementation without
  hanging: under the OLD loop, the mocked `_ssh_capture` payload makes
  `_slurm_state` return a truthy string forever and `time.sleep` is a no-op,
  so the loop never exits — which is precisely the missing-watchdog failure
  this task fixes. An initial run with the brief's full `-k` expression hung
  and was killed; the fast-failing pair above is the recorded red step.
- Step 4 (green): `uv run python scripts/dev.py test tests/test_remote.py` →
  **122 passed in 2.38s** (full file: all four new tests plus every updated
  pre-existing attach/stop test).
- `uv run python scripts/dev.py lint` → "All checks passed!";
  `uv run python scripts/dev.py typecheck` → "0 errors, 0 warnings,
  0 informations".

## Deviations from the brief

1. Red-step `-k` narrowed to `"poll_chain or stop_sentinel"` because the
   watchdog/slice-gap tests hang (infinite loop) against the old
   implementation — see TDD evidence.
2. Removed the now-dead `_read_progress_records` helper. Not requested, but
   its only caller was replaced and an unreferenced private ssh helper seemed
   worse than the 10-line deletion. Trivial to restore if a strictly minimal
   diff is preferred.
3. The log-stream watchdog threshold is interpolated from `_POLL_GRACE_POLLS`
   (f-string) rather than a literal 15, so both viewers share one constant.

## For the reviewer to scrutinize

- `_poll_chain`'s remote command embeds `_squeue_state_command(..., retired="STATE=")`
  inside a `case` arm; a non-"Invalid job id" squeue failure still `exit`s the
  ssh command nonzero, which `_ssh_capture` turns into SystemExit — same
  fail-closed behavior as `job_status`/`_live_jobs`, but now on the attach
  poll path (a transient squeue outage kills the viewer rather than counting
  as a miss). This matches the brief's sketch verbatim.
- The dashboard's `missed` counter counts any non-terminal poll with no live
  SLURM job; a job PENDING in squeue counts as live, so long queue waits do
  not trip the watchdog.
- `test_stop_writes_stop_sentinel_before_scancel` uses `cmd.index("STOP")`;
  the first "STOP" occurrence in the command is the sentinel path (nothing
  earlier contains that substring — the jobid is `20260717-abc`).
- The uncommitted working-tree modifications to
  `.superpowers/sdd/task-3-report.md` and `task-4-report.md` predate this task
  and were left untouched.
