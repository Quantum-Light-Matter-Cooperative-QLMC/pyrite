---
name: remote-gpu-jobs
description: Use when preparing sweeps, heavy Monte Carlo, or GPU-bound PyRITE work; route compute to lab box through pyrite remote instead of local WSL.
---

# Remote GPU Jobs

Never run heavy sweep locally; WSL multiprocessing can OOM/crash.

1. Run `pyrite remote sync` only when an explicit standalone sync is needed;
   submission syncs by default.
2. Submit with `pyrite run [PROFILE] --remote`; use `-m MATERIAL` for one member.
3. Observe with `pyrite job status`, `pyrite job logs --follow`, or `pyrite job attach`.
4. Pull results with `pyrite remote pull <profiles-or-stems...>`.
5. Cancel with `pyrite job stop <job-id>`; later submission resumes checkpoints.

Default ~10-minute SLURM chunks provide scheduler yield points and checkpoint
resume; override with `pyrite run ... --remote --chunk-minutes N` (remote only). `--chunk-minutes 0` monopolizes one allocation;
use only with confirmed box availability.
