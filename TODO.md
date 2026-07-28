# TODO — feature/remote-ux

Remote-job UX fixes around `cxr remote submit/stop/attach`: profile-aware job
naming, quieter submit output, faster/safer stop, and attach-time job control
plus compute-aware progress. Items drafted by user in the cli-profile-rework
worktree TODO (2026-07-28); every CLI change goes through the `cli-ui-ux`
skill.

## Items

1. **Profile-based job names + same-profile submit block.** Job ids like
   `20260727-235609-0145591e` are unreadable; name jobs from the profile
   (e.g. `<profile>` or `<profile>-<n>`), block a second submit under the
   same profile while one is live (1 of a profile queued at a time; no
   name-conflict handling needed).
2. **Submit pull-suggestion text-vomit.** `cxr remote submit --all --profile
   sub_100keV` prints a 20-material `cxr remote pull <hash-stem> ...` line
   nobody can type. Replace with the `MATERIAL@PROFILE` pull form (or a
   per-profile/wildcard pull), wrapped/grouped output.
3. **Stop lists only explicit materials.** `stop` should name specific mats
   only when they were explicitly submitted as materials, not when they came
   from `--all`/a profile.
4. **Speed up `cxr remote stop --all --yes`.** 30+ s to quit jobs; likely
   serial SSH/scancel per job — batch or parallelize.
5. **Cancel SLURM job while attached.** Keybinding in `cxr remote attach`
   to scancel the attached job (a couple keystrokes, confirm; not bare `q`).
6. **Compute-aware progress in attach.** Port the compute-progress
   calculations from `notebooks/scan_app.py` to the remote progress bars:
   replace the top-level summary bar in plain `cxr remote attach`, show both
   bars under `-v/-vv`. (Folds backlog P2 "remote progress readout" item.)

## Order

3 → 2 → 1 (submit/stop/naming cluster), then 4, then 5/6 (attach cluster).
