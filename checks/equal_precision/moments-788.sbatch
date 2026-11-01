#!/bin/bash
#SBATCH --job-name=moments788
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:08:00
#SBATCH --output=moments788-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec 9>.pyrite-sync.lock
flock -s -w 30 9
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog"
export PYRITE_MC_BACKEND=cuda PYRITE_TEST_BACKEND=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
out="evidence/moments788/$SLURM_JOB_ID"
mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"
scontrol show job "$SLURM_JOB_ID" > "$out/job.txt"
nvidia-smi -q > "$out/gpu.txt"
uv run --no-sync pyrite-dev test tests/montecarlo/cascades/test_radiative_moments_cuda.py \
    --durations=0 > "$out/cuda-tests.log" 2>&1
uv run --no-sync python checks/equal_precision/qualify_moments.py "$out/components.json"
