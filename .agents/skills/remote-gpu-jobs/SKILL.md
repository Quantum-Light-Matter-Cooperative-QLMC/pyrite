---
name: remote-gpu-jobs
description: Use when about to run a sweep, heavy Monte-Carlo compute, or any GPU-bound cxr-mc workload -- routes it to the lab box via cxr remote instead of running locally.
---

# Remote GPU jobs

**Never run a sweep on the laptop.** Local multiprocessing sweeps crash WSL.
All heavy Monte-Carlo compute belongs on the lab GPU box (`qlmc`), submitted
through `cxr remote`.

## Workflow

1. `cxr remote sync` -- push the current working tree (start/scan do this
   automatically unless `--no-sync`).
2. `cxr remote start <materials...>` (or `--all`) -- submit and walk away.
   Survives ssh disconnect.
3. `cxr remote status` / `logs --follow` / `attach` -- observe any time.
4. `cxr remote pull <stems...>` when state is `done`.

## Chunking (the default)

`start`/`scan` submit a self-resubmitting chain of ~10-minute SLURM slices
(`--chunk-minutes`, default 10). Each slice boundary is a scheduler decision
point and every slice carries `--nice=10000`, so other users' jobs jump ahead
roughly every 10 minutes. The sweep still runs unattended to completion:
slices resume from the per-material checkpoint.

- `--chunk-minutes 0` deliberately takes the whole box in one allocation.
  Only do this when nobody else needs it.
- To free the box for someone: `cxr remote stop <material>`. The chain
  cancels cleanly (no orphan resubmission) and a later `start` resumes from
  checkpoint with nothing lost beyond the in-flight config.
