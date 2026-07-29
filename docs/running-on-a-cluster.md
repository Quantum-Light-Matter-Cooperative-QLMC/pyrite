# Running on a cluster (SLURM)

`cxr scan` is a headless entry point: it runs one material's Monte-Carlo sweep and
writes `checkpoints/<material>/{line,brem}.pkl`. That makes it a clean fit for any
batch scheduler without requiring the optional lab-box helper described below.
Install the package once on the cluster, submit one job per material, then pull
the checkpoints back and do interactive analysis or static-HTML export locally.

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
uv run cxr scan "$MATERIAL"      # -> checkpoints/<material>/{line,brem}.pkl
uv run cxr scan "$MATERIAL" --profile survey  # reduced identity-qualified variant
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
uv run cxr scan "${MATERIALS[$SLURM_ARRAY_TASK_ID]}"
```

## 4. Retrieve and visualize locally

The checkpoints are the only output you need off the cluster:

```bash
rsync -avz login-node:~/cxr-mc/checkpoints/ ./checkpoints/
```

Then run `cxr analyze <material>` (the `notebooks/analysis_app.py` marimo app) or
run `cxr export` locally —
all interactive visualization and static-HTML export stay on your workstation.

## Notes

- **`__main__` guard:** `cxr scan` (and the `python -m cxr_mc._entry.scan` shim) are properly
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
cxr remote start hopg --dry-run
```

`cxr remote submit hopg` syncs, submits, follows the SLURM job, and pulls the
checkpoint. Add `--headless` to return immediately after submission.
Use `--chunk-minutes 0 --parallel-materials 3` only for workloads measured to
fit concurrently. Use `cxr remote status`, `cxr remote logs --follow`,
or `cxr remote attach` to monitor the allocation. `attach` shows an independent
case-progress bar for each material; `logs --follow` shows the raw shared job log.

Use `--performance-profile NAME` to run existing catalog profile `NAME` with
resource sampling enabled; it supplies the same profile selection as
`--profile NAME`, so do not repeat both options:

```bash
cxr remote submit --performance-profile sub_100keV
```

Local `cxr scan MATERIAL --performance-profile NAME` uses the same resolution.
Each five-second NDJSON sample records host and process-tree CPU/RAM, CPU
affinity/frequency/iowait, swap activity, GPU utilization, clocks, performance
state, and VRAM (`null` when no NVIDIA GPU is available), power, temperature,
resolved beam parameters, active driver phase/case, most recently completed
case, in-flight work, progress, worker topology, electron counts, grid widths,
adaptive spectrum/brem chunk sizes, and child-process max/mean RSS. Rolling
counters include CPU transport, spectrum, GPU feed-wait, checkpoint time, GPU
OOM retries, and CuPy pool used/reserved/peak memory. These phase counters are
enabled by
`--performance-profile`; `CXR_MC_TIMING` is not required. Local logs go to
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
`cxr remote stop ...` cancels an active allocation with `scancel`. `cxr remote
check` follows the same submit-and-wait workflow for its validation calculation;
`cxr remote check --detached` returns after submission.
