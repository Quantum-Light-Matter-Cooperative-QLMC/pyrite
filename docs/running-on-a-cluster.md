# Running on a cluster (SLURM)

`cxr scan` is a headless entry point: it runs one material's Monte-Carlo sweep and
writes a single `checkpoints/<material>.pkl`. That makes it a clean fit for any
batch scheduler — there is no bespoke remote-execution tool to adopt. Install the
package once on the cluster, submit one job per material, then pull the
checkpoints back and do all the (matplotlib / PDF) visualization locally.

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
uv run cxr scan "$MATERIAL"      # -> checkpoints/<material>.pkl
```

On a GPU node `run_cases` runs serially (one CUDA context), so `--cpus-per-task`
mainly helps the CPU fallback. For a **CPU-only** partition, drop `--gres` and the
CUDA module and pass `--workers $SLURM_CPUS_PER_TASK` to use the transport pool.

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
all the matplotlib / webpdf work stays on your workstation, where that toolchain
lives.

## Notes

- **`__main__` guard:** `cxr scan` (and the `python scan.py` shim) are properly
  guarded, so the `spawn` / `forkserver` transport workers are safe. Don't wrap the
  sweep in an unguarded `python -c "…"`.
- **`--quick`** runs a tiny smoke grid into `<material>_quick.pkl` — use it to
  validate your sbatch script cheaply before submitting the full sweep.
- **fp64:** set `CXR_FP64=1` for double-precision reference runs (the GPU path
  defaults to fp32).

## Lab-box remote helper

`cxr remote` syncs the current working tree to the configured lab host and
submits every CXR compute run through SLURM. Its fixed lab allocation requests
the `gpu` partition, one node, one task, one GPU (`--gres=gpu:1`), and unlimited
wall time. The generated batch job starts with `module purge`, then loads
`cuda`, `openmpi`, and `hdf5`; it uses the synced project's configured `uv`
environment, not the WarpX-specific `jrozells` Conda environment.

Review the exact batch script and `sbatch --parsable` submission command without
contacting the lab box:

```bash
cxr remote start hopg --dry-run
```

`cxr remote scan hopg` syncs, submits, follows the SLURM job, and pulls the
checkpoint. `cxr remote start hopg wse2` syncs, submits, and returns immediately.
Multi-material runs start two material scans concurrently by default; use
`--parallel-materials 3` for workloads measured to fit concurrently, `1` for
serial execution, or at most `4`. Use `cxr remote status`, `cxr remote logs --follow`,
or `cxr remote attach` to monitor the allocation. `attach` shows an independent
case-progress bar for each material; `logs --follow` shows the raw shared job log.
`cxr remote stop ...` cancels an active allocation with `scancel`. `cxr remote
check` follows the same submit-and-wait workflow for its validation calculation;
`cxr remote check --detached` returns after submission.
