# TODO — feature/remote-ux

Remote-job UX fixes around `cxr remote submit/stop/attach`: profile-aware job
naming, quieter submit output, faster/safer stop, and attach-time job control
plus compute-aware progress. Items drafted by user in the cli-profile-rework
worktree TODO (2026-07-28); every CLI change goes through the `cli-ui-ux`
skill.

## Items

1. ~~**Profile-based job names + same-profile submit block.**~~ [DONE]
   Profile submits name the job `NAME` (first free `NAME-N` once a finished
   run holds the bare name; `state._profile_jobdirs`); a submit under a
   profile with a live job is refused with attach/stop guidance
   (`lifecycle._refuse_if_profile_live`). Standard/explicit-material submits
   keep timestamp ids.
2. ~~**Submit pull-suggestion text-vomit.**~~ [DONE] `pull --profile NAME`
   expands to each member's `MATERIAL@PROFILE` selector (explicit MATERIALs
   qualified the same way; `--all` rejected); profile submits now print
   `cxr remote pull --profile NAME (after completion)` instead of the
   hash-stem wall.
3. ~~**Stop lists only explicit materials.**~~ [DONE] `stop --profile NAME`
   stops live jobs whose recorded `catalog_profile` metadata matches
   (`state._job_profiles`); material names stay the handle only for
   explicit-material submits.
4. ~~**Speed up `cxr remote stop --all --yes`.**~~ [DONE] one batched
   `scripts._scancel_jobs_command`: all STOP sentinels first, single scancel,
   one shared squeue poll, per-job release + terminal state; skips jobs with
   no scheduler id. Single-job callers keep `_stop_jobid`.
5. ~~**Cancel SLURM job while attached.**~~ [DONE] `viewer._KeyListener`:
   background nonblocking single-keypress capture, POSIX `termios`/`tty`
   cbreak + `select`, Windows `msvcrt` (picked via `try: import msvcrt` /
   `except ImportError`, not a `sys.platform` literal check, so static
   analysis doesn't flag the POSIX branch unreachable). Cancelling needs TWO
   different keys -- `x` arms, `y` confirms within `_CANCEL_ARM_SECONDS`
   (6s); any other key or a lapsed window disarms silently. Confirmed cancel
   calls `lifecycle._stop_jobid`; a no-op (no hint shown) when stdin isn't a
   tty. `_live_status` return value unchanged (False on cancel, same as
   disconnect/stall) so scan/check auto-pull logic didn't need touching.
6. ~~**Compute-aware progress in attach.**~~ [DONE] `sweep.case_cost` now
   threads through `run_sweep(case_cost_fn=..., on_cost=...)` (run.py) ->
   `scan._run_material` -> the progress JSON's new `done_cost`/`total_cost`
   fields (scan.py's `_write_progress_record`, additive/optional so
   rebrem/reline/blaze keep writing the old shape). `presentation.py`'s
   `_overall_progress_line(..., use_cost=True)` and `_format_job_status`
   render ONE compute-weighted bar at base verbosity when cost data exists
   (falls back to the legacy case-count bar otherwise, byte-identical to
   before), both bars side by side under `-v/-vv`.

## Landed alongside

- `sub_100keV` ships an explicit materials membership (seeded
  `add-material --all`), so `submit --profile sub_100keV` needs no
  MATERIAL/--all; `add-material` gained `-a` short flag. `profile create`
  already clones standard's beam-energy defaults.

## Order

3 → 2 → 1 (submit/stop/naming cluster), then 4, then 5/6 (attach cluster).
All six items done.
