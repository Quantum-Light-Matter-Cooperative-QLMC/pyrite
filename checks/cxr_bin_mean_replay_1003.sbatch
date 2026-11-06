#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o scratch/bin1003/replay-%j.log
# #1003: replay captured bin-mean flushes (arguments: .npz fixtures) through
# auto/pairs/tree with and without the midpoint series, then the CUDA tests.
# Run from a clean isolated checkout at the pushed SHA.
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/scratch/bin1003/empty-user-catalog"
export XDG_CONFIG_HOME="$PWD/scratch/bin1003/xdg"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
export PYRITE_MC_BACKEND=cuda PYRITE_TEST_BACKEND=cuda PYRITE_FP64=0
export OPENBLAS_NUM_THREADS=1
mkdir -p "$PYRITE_USER_CATALOG" "$XDG_CONFIG_HOME"
git rev-parse HEAD
git status --short
nvidia-smi --query-gpu=name,driver_version --format=csv
uv sync -q --locked --no-default-groups --extra nvidia --group dev
uv run --no-sync pyrite-dev test tests/montecarlo/spectrum/test_sinc_bin_integration_cuda.py \
  tests/montecarlo/spectrum/test_sinc_bin_near_far.py || echo "FAILED cuda tests rc=$?"
arms="${ARMS:-auto,pairs,tree,auto+series,pairs+series,tree+series}"
for fixture in "$@"; do
  name="$(basename "$fixture" .npz)"
  uv run --no-sync python checks/cxr_bin_mean_profile_1003.py replay "$fixture" \
    --arms "$arms" --repeats "${REPEATS:-3}" --out "scratch/bin1003/replay2-$name.json"
done
echo DONE
