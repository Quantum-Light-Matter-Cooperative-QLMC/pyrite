#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o scratch/bin1003/replay-%j.log
# #1003: replay captured bin-mean flushes (arguments: .npz fixtures) through
# the reducer arms with and without the midpoint series, after the CUDA tests,
# and optionally Nsight Compute (NCU_FIXTURE) on a slice.
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
arms="${ARMS:-auto,pairs,tree,sparse,sparse-line,auto+series,pairs+series,tree+series,sparse-line+series}"
for fixture in "$@"; do
  name="$(basename "$fixture" .npz)"
  uv run --no-sync python checks/cxr_bin_mean_profile_1003.py replay "$fixture" \
    --arms "$arms" --repeats "${REPEATS:-2}" --out "scratch/bin1003/replay2-$name.json"
done
if [ -n "${NCU_FIXTURE:-}" ]; then
  ncu --section SpeedOfLight --section Occupancy --section ComputeWorkloadAnalysis \
    --section WarpStateStats --section LaunchStats \
    --kernel-name regex:'pyrite_bin_tree_reduce|sparse_near' --launch-count 12 \
    uv run --no-sync python checks/cxr_bin_mean_profile_1003.py replay "$NCU_FIXTURE" \
      --out scratch/bin1003/replay2-ncu.json --slice-lines 50000 --max-flushes 1 --repeats 1 \
      --arms auto,auto+series,sparse-line+series,sparse-line \
    > scratch/bin1003/ncu2.txt 2>&1 || echo "FAILED ncu rc=$?"
fi
echo DONE
