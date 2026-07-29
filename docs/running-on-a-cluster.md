# Running on a cluster (SLURM)

`cxr run [PROFILE] -m MATERIAL` is the headless entry point for one profile
member. It writes `checkpoints/<material>/{line,brem}.pkl`, making it a clean
fit for any batch scheduler without the optional lab-box helper below. Install
once, submit one job per material, then pull checkpoints back for local
analysis or static-HTML export.

> The scripts below are **templates** — partition names, the CUDA module, account
> strings, and resource limits are site-specific. Adapt them to your cluster.

## 1. Install on the cluster

The project is uv-managed with a committed lockfile. On the login node:

```bash
git clone https://github.com/Quantum-Light-Matter-Cooperative-QLMC/cxr-mc.git
cd cxr-mc
uv sync                       # .venv + locked deps + the cxr_mc package
uv run cxr --help             # sanity check
```

No GPU is required to install — `cupy` imports cleanly and the code falls back to
CPU automatically. For GPU runs the compute node needs a CUDA runtime matching the
`cupy-cuda13x` wheel (load it with `module load cuda/13.x` or similar).

## 2. One material per job

Submit with `sbatch run_cxr.sh mose2`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=cxr-scan
#SBATCH --partition=gpu          # <-- your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=cxr-%x-%j.out

set -euo pipefail
module load cuda/13.x            # <-- match the cupy-cuda13x wheel (omit for CPU)
cd "$SLURM_SUBMIT_DIR"

MATERIAL="${1:?usage: sbatch run_cxr.sh <material>}"
uv run cxr run standard -m "$MATERIAL"
uv run cxr run standard -m "$MATERIAL" --fidelity survey
```

On a GPU node one main-process CUDA context handles spectrum/bremsstrahlung while
a process pool prepares CPU electron transport, so `--cpus-per-task` supplies
those transport workers. For a **CPU-only** partition, drop `--gres` and the CUDA
module; `run_cases` uses a full-case worker pool capped by both core count and
available memory. Pass `--workers $SLURM_CPUS_PER_TASK` to request the allocation's
CPU count; the memory cap still applies.

## 3. Several materials as a job array

One array task per material — they run independently and write their own pickles:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=cxr-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-3              # <-- 0..N-1 for the N materials below
#SBATCH --output=cxr-%x-%A_%a.out

set -euo pipefail
module load cuda/13.x
cd "$SLURM_SUBMIT_DIR"

MATERIALS=(mose2 wse2 mos2 hopg)            # indexed by $SLURM_ARRAY_TASK_ID
uv run cxr run standard -m "${MATERIALS[$SLURM_ARRAY_TASK_ID]}"
```

## 4. Retrieve and visualize locally

The checkpoints are the only output you need off the cluster:

```bash
rsync -avz login-node:~/cxr-mc/checkpoints/ ./checkpoints/
```

Then run `cxr app analysis <material>` (the `notebooks/analysis_app.py` marimo app) or
run `cxr app analysis export` locally —
all interactive visualization and static-HTML export stay on your workstation.

## Notes

- **`__main__` guard:** `cxr run` (and the `python -m cxr_mc._entry.scan` shim) are properly
  guarded, so the `spawn` / `forkserver` transport workers are safe. Don't wrap the
  sweep in an unguarded `python -c "…"`.
- **`--quick`** runs a tiny smoke grid into `<material>_quick.pkl` — use it to
  validate your sbatch script cheaply before submitting the full sweep.
- **fp64:** set `CXR_FP64=1` for double-precision reference runs (the GPU path
  defaults to fp32).

## Lab-box remote helper

`cxr remote` syncs the current working tree to the configured lab host and
submits every CXR compute run through SLURM. Its fixed lab allocation requests
the `gpu` partition, one node, one task, and one GPU (`--gres=gpu:1`). Default
`--chunk-minutes 10` runs one material at a time in bounded, self-resubmitting
slices. `--chunk-minutes 0` selects a monolithic `UNLIMITED` allocation; only
that mode accepts `--parallel-materials`, defaults to one material,
and caps concurrency at four. The generated batch job starts with `module purge`, then loads
`cuda`, `openmpi`, and `hdf5`; it uses the synced project's configured `uv`
environment, not the WarpX-specific `jrozells` Conda environment.

Review the exact batch script and `sbatch --parsable` submission command without
contacting the lab box:

```bash
cxr remote run standard -m hopg --dry-run
```

`cxr remote run standard -m hopg` syncs, submits, follows the SLURM job, and
pulls the checkpoint. Add `--headless` to return after submission.
Use `--chunk-minutes 0 --parallel-materials 3` only for workloads measured to
fit concurrently. Use `cxr remote status`, `cxr remote logs --follow`,
or `cxr remote attach` to monitor the allocation. `attach` shows an independent
case-progress bar for each material; `logs --follow` shows the raw shared job log.

Use `--perf` to enable resource sampling for the selected profile:

```bash
cxr remote run sub_100keV --perf
```

Each five-second NDJSON sample records host and process-tree CPU/RAM, CPU
affinity/frequency/iowait, swap activity, GPU utilization, clocks, performance
state, and VRAM (`null` when no NVIDIA GPU is available), power, temperature,
resolved beam parameters, active driver phase/case, most recently completed
case, in-flight work, progress, worker topology, electron counts, grid widths,
adaptive spectrum/brem chunk sizes, and child-process max/mean RSS. Rolling
counters include CPU transport, spectrum, GPU feed-wait, checkpoint time, GPU
OOM retries, and CuPy pool used/reserved/peak memory. These phase counters are
enabled by `--perf`; `CXR_MC_TIMING` is not required. Logs go to
`performance-profiles/NAME/<material>.ndjson`; remote logs appear beside case
progress in `attach`. Fetch every remote job matching the catalog profile name
with:

```bash
cxr remote profile pull sub_100keV
```

Pulled files land under `performance-profiles/NAME/<job>/<material>.ndjson`.
Use the
[performance-profile analysis playbook](performance-profile-analysis.md) to
validate sessions, derive phase/resource metrics, classify bottlenecks, and
design controlled tuning runs.

For a bursty, spectrum-dominated GPU run, add `--nsys --chunk-minutes 0` to a
single-material, single-repetition performance submit. This runs an uncached
job-local session under Nsight Systems and writes CUDA/NVTX/Python-stack trace
artifacts beside the NDJSON. `cxr remote profile pull NAME` fetches the
`.nsys-rep`, `.sqlite`, and `.nsys-stats.txt` files too; see the playbook's
Nsight section for the exact command and interpretation limits.

`cxr remote stop ...` cancels an active allocation with `scancel`. `cxr remote
check` follows the same submit-and-wait workflow for its validation calculation;
`cxr remote check --detached` returns after submission.

With no job id, `cxr remote attach`/`logs`/`status` resolve to the most
recently active job. Profile submissions name their job after the profile
(`sub_100keV`, then `sub_100keV-2` once the bare name is taken), so
resubmitting a profile leaves the earlier, now-terminal jobs on the box.
`attach` warns on stderr when it defaults to a job that is no longer running,
so a stale default never masks the live resubmission.

`cxr remote prune-jobs` deletes terminal (done/failed/cancelled) job
directories, previewing exact targets unless `--yes`. Scope it to one profile
family with `--profile NAME` or sweep every profile with `--all`. A live chain
is never removed, so it is safe to prune old runs of a profile while a fresh
submission of the same profile is still going:

```bash
cxr remote prune-jobs --profile sub_100keV        # preview
cxr remote prune-jobs --profile sub_100keV --yes  # delete
```
