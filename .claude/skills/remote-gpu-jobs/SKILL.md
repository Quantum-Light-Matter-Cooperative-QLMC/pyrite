---
name: remote-gpu-jobs
description: Use when preparing sweeps, heavy Monte Carlo, or GPU-bound cxr-mc work; route compute to lab box through cxr remote instead of local WSL.
---

# Remote GPU Jobs

Never run heavy sweep locally; WSL multiprocessing can OOM/crash.

1. `cxr remote sync` when explicit sync needed; submit syncs by default.
2. `cxr remote run [PROFILE]`; use `-m MATERIAL` for one profile member.
3. Observe with `status`, `logs --follow`, or `attach`.
4. `cxr remote pull <stems...>` after completion.
5. `cxr remote stop <material>` to free box; later run resumes checkpoint.

Default ~10-minute SLURM chunks (`--chunk-minutes`) provide scheduler yield
points and checkpoint resume. `--chunk-minutes 0` monopolizes one allocation;
use only with confirmed box availability.
