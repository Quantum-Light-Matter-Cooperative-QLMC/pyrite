---
name: remote-gpu-jobs
description: Use when preparing sweeps, heavy Monte Carlo, or GPU-bound cxr-mc work; route compute to lab box through cxr remote instead of local WSL.
---

# Remote GPU Jobs

Never run heavy sweep locally; WSL multiprocessing can OOM/crash.

1. Run `cxr remote sync` only when an explicit standalone sync is needed;
   submission syncs by default.
2. Submit with `cxr run [PROFILE] --remote`; use `-m MATERIAL` for one member.
3. Observe with `cxr job status`, `cxr job logs --follow`, or `cxr job attach`.
4. Pull results with `cxr remote pull <profiles-or-stems...>`.
5. Cancel with `cxr job stop <job-id>`; later submission resumes checkpoints.

Default ~10-minute SLURM chunks (`--chunk-minutes`) provide scheduler yield
points and checkpoint resume. `--chunk-minutes 0` monopolizes one allocation;
use only with confirmed box availability.
