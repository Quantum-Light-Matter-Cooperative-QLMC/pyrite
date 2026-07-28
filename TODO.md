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
5. **Cancel SLURM job while attached.** Keybinding in `cxr remote attach`
   to scancel the attached job (a couple keystrokes, confirm; not bare `q`).
   Note: viewer runs `ssh -n` with stdin closed and is Windows-careful —
   keypress handling needs cross-platform nonblocking stdin (msvcrt/termios).
6. **Compute-aware progress in attach.** Port the compute-progress
   calculations from `notebooks/scan_app.py` to the remote progress bars:
   replace the top-level summary bar in plain `cxr remote attach`, show both
   bars under `-v/-vv`. (Folds backlog P2 "remote progress readout" item.)

## Landed alongside

- `sub_100keV` ships an explicit materials membership (seeded
  `add-material --all`), so `submit --profile sub_100keV` needs no
  MATERIAL/--all; `add-material` gained `-a` short flag. `profile create`
  already clones standard's beam-energy defaults.

## Order

3 → 2 → 1 (submit/stop/naming cluster), then 4, then 5/6 (attach cluster).
