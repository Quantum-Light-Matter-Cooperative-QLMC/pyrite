#!/bin/bash
#SBATCH --job-name=probe803-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --nodelist=qlmc-queen
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=probe803-cpu-%A_%a.log
# #803 CPU component probe: one array task per scene, one thread, fresh process per sample.
# Pinned to the 9950X3D CPU node (same model as the GPU host and #792 records).
# Resumable: re-submit with the same PROBE803_OUT to continue a partial directory.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
BR="$PWD"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/tmp/pyrite-uv-cache
export PYRITE_MC_BACKEND=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYRITE_HOME="$BR" PYRITE_CATALOG="$BR/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$BR/user-catalog"
export XDG_CONFIG_HOME="$BR/config-home" XDG_DATA_HOME="$BR/data-home"; mkdir -p "$XDG_CONFIG_HOME"
scenes=(si-300-100um w-300-10um sio2-800-tr0.3 w2um-on-si100um-800)
scene="${scenes[$SLURM_ARRAY_TASK_ID]}"
out="${PROBE803_OUT:-$BR/evidence/probe803/cpu-$SLURM_ARRAY_JOB_ID}"; mkdir -p "$out"
{ git rev-parse HEAD 2>/dev/null || cat .git/HEAD; } > "$out/head-$scene.txt"
scontrol show job "$SLURM_JOB_ID" > "$out/job-$scene.txt"
uv run --no-sync python checks/equal_precision/run_probe_803.py "$out" \
    --core per-electron --scene "$scene" --timeout-s 3600
