#!/bin/bash
#SBATCH --job-name=lut803-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%1
#SBATCH --output=lut803-gpu-%A_%a.log
# #803 CUDA baseline: scenes serialized on one GPU (%1), fresh process per sample.
# Resumable: re-submit with the same LUT803_OUT to continue a partial directory.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
BR="$PWD"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/tmp/pyrite-uv-cache
export PYRITE_MC_BACKEND=cuda OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYRITE_HOME="$BR" PYRITE_CATALOG="$BR/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$BR/user-catalog"
export XDG_CONFIG_HOME="$BR/config-home" XDG_DATA_HOME="$BR/data-home"; mkdir -p "$XDG_CONFIG_HOME"
unset PYRITE_MC_TRANSPORT_SCHEDULE PYRITE_MC_ELECTRON_BATCH_MAX
scenes=(si-300-100um w-300-10um sio2-800-tr0.3 w2um-on-si100um-800)
scene="${scenes[$SLURM_ARRAY_TASK_ID]}"
out="${LUT803_OUT:-$BR/evidence/lut803/gpu-$SLURM_ARRAY_JOB_ID}"; mkdir -p "$out"
{ git rev-parse HEAD 2>/dev/null || cat .git/HEAD; } > "$out/head-$scene.txt"
scontrol show job "$SLURM_JOB_ID" > "$out/job-$scene.txt"; nvidia-smi -q > "$out/gpu-$scene.txt"
uv run --no-sync python checks/equal_precision/run_baseline_803.py "$out" \
    --core cuda --scene "$scene" --timeout-s 2400
