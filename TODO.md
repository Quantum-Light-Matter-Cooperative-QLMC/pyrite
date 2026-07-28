# TODO / Backlog — feature/remote-attach-ux

User-reported `cxr remote` attach / progress / submit UX problems, folded from
a `>user<` item on `main` (2026-07-28, via `feature/cli-fixes` split).
Original user text preserved verbatim; triage notes inline.

Overlap note: `main` P2 "Remote-job UX" (planned `feature/remote-ux`) covers
attach-time cancel key + compute-aware progress bars. Coordinate scopes when
that branch is created; items below are the authoritative user text for the
progress-bar and submit-merge work.

## User text

1. Progress bars (both 'compute' and 'cases') still only show the number of cases for the materials current displayed. Example: sub_100keV has 21 mats, 1568 cases per mat, starts with HOPG, then hBN. I submit, then run `cxr remote attach sub_100keV`. the summary progress bar *labels* both then show `829/1568 cases`, though the actual bar fill seems correct.
   1. For the `cases` bar, this should be fixed to be `829/(total_cases)`
   2. For `compute`, it should just be a percentage of completed compute.
   3. We don't need the repeated `x/21 materials` fields; that can just be its own line reported once, which should be shown on all levels of verbosity.
   4. For all these scan-progression indicators on `attach` or otherwise, it would be good on high-verbosity modes to report (with more colored bars but different colorscheme) showing compute usage -- CPU utilization, GPU utilization, percent/absolute memory utilization for host & GPU VRAM, etc. Those can be only on the highest verbosity level, though.
   5. When `cxr remote attach` gets paused/queued, the job-wide pregress bar(s) go green with a checkmark as if they're done. They should turn orange and go to the paused state. Same with the 'State' and 'SLURM' texts--they go blue when 'PENDING/queued' rather than the paused color (also, do we really need both of those status indicators? Seems just one would do--the state one, and say 'PENDING' rather than 'queued')
   6. 'profile=sub_100keV' should go on its own line
2. Add info on whether or not chi_g/U_g were pulled from cache or recomputed for each mat on the highest verbosity level
3. `remote attach` often gets disconnected by remote host closing connection. Add error handling to try to reconnect a couple times before allowing disconnect with explanatory message
4. Add some `squeue` information dipslay reports tracking SLURM progress
5. `remote attach`, add `p -> y (confirm)` sequence to pull the (potentially partially completed) items from the current job/profile being tracked
6. `remote scan` and `remote submit` should be combined into one `remote submit` which defaults to `scan`'s behavior, with a `--headless` flag to do current `submit behavior` and a `--no-pull` flag (can't do both since `--headless` won't pull anyway) which will attach & track progress but won't auto-pull results
7. `remote clear` should be able to clear based on `--profile`
8. evaluate `cxr [remote] prune [--all OR --profile NAME]` to drop stale checkpoints locally or on remote. Prune shouldn't require confirmation

## Triage (suggested order)

- B1 (bug, quick wins): 1.1, 1.2, 1.3, 1.6 — progress-bar labels/denominators and layout fixes.
- B2 (bug): 1.5 — paused/queued state rendering (orange paused, not green done; single `State` indicator reading `PENDING`).
- B3 (feature): 3 — reconnect-with-retry on attach disconnect.
- B4 (feature, breaking): 6 — merge `remote scan` into `remote submit` with `--headless` / `--no-pull`. Deprecate `scan` per `cli-ui-ux`.
- B5 (feature): 7, 8 — `clear --profile`, evaluate `prune`.
- B6 (feature, high-verbosity only): 1.4, 2, 4 — compute-usage bars, chi_g/U_g cache provenance, `squeue` display.
- B7 (feature): 5 — `p -> y` interactive partial pull in attach.
