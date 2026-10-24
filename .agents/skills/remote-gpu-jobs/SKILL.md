---
name: remote-gpu-jobs
description: Use when preparing sweeps, heavy Monte Carlo, or GPU-bound PyRITE work; route compute to lab box through pyrite remote instead of local WSL.
---

# Remote GPU Jobs

Never run heavy sweep locally; WSL multiprocessing can OOM/crash.

## Pre-flight

Run before every submission; most failed remote runs were stale state.

1. Checkpoint first: with push authority, `ship checkpoint` so a session limit
   cannot strand the code behind a running job.
2. Code identity: commit the code under test and record `git rev-parse HEAD`
   in the job name or log. Sync ships the working tree, so run from a clean
   tree. `pyrite remote sync` skips `tests/`; for remote pytest, also
   `rsync -az --delete --exclude __pycache__ --delete-excluded tests/ <host>:<remote_dir>/tests/`.
3. Scheduler: `ssh <host> sinfo -p gpu`; do not submit to a `drain`/`down`
   node (`--exclude=<node>`). A job pending for minutes is checked with
   `squeue`/`sacct`, not waited on.
4. Ad-hoc `sbatch` scripts (not `pyrite run --remote`) must
   `export PATH=$HOME/.local/bin:$PATH` (SLURM batch env lacks `uv`), set
   `PYRITE_HOME`, `PYRITE_CATALOG`, and `PYRITE_USER_CATALOG` to the synced
   checkout paths as `pyrite.remote.config.remote_runtime_env()` does (the box
   config pins a stale catalog), and set `--mem`. The `gpu` partition rejects
   `--exclusive`.
5. Record job IDs in the issue's `## Progress` section and keep working; post
   results when the job ends. To collect results without a live session, a
   headless follow-up works:
   `claude -p "Collect results of job <id> for issue #<n> and update its Progress section"`.

## Run

1. Run `pyrite remote sync` only when an explicit standalone sync is needed;
   submission syncs by default.
2. Submit with `pyrite run [PROFILE] --remote`; use `-m MATERIAL` for one member.
3. Observe with `pyrite job status`, `pyrite job logs --follow`, or `pyrite job attach`.
4. Pull results with `pyrite remote pull <profiles-or-stems...>`.
5. Cancel with `pyrite job stop <job-id>`; later submission resumes checkpoints.

Default ~10-minute SLURM chunks provide scheduler yield points and checkpoint
resume; override with `pyrite run ... --remote --chunk-minutes N` (remote only). `--chunk-minutes 0` monopolizes one allocation;
use only with confirmed box availability.
