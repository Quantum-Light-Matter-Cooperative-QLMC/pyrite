#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o scratch/bin1003/profile-%j.log
# #1003: CUDA tests, a CUDA-transport capture of the h-BN 5 MeV case on the
# frozen grid, every reducer arm replayed on it and on the CPU capture, and an
# Nsight Compute pass on a slice. Run from a clean isolated checkout at the
# pushed SHA, with scratch/bin1003/{grid-hbn-seed0.npy,hbn5mev-cpu-n128.npz}.
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/scratch/bin1003/empty-user-catalog"
export XDG_CONFIG_HOME="$PWD/scratch/bin1003/xdg"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
export PYRITE_MC_BACKEND=cuda PYRITE_TEST_BACKEND=cuda
export PYRITE_FP64=0 PYRITE_MC_TRANSPORT_PRECISION=fp64
export OPENBLAS_NUM_THREADS=1
mkdir -p "$PYRITE_USER_CATALOG" "$XDG_CONFIG_HOME"
out=scratch/bin1003
git rev-parse HEAD
git status --short
nvidia-smi
uv sync -q --locked --no-default-groups --extra nvidia --group dev
uv run --no-sync pyrite-dev test tests/montecarlo/spectrum/test_sinc_bin_integration_cuda.py \
  tests/montecarlo/spectrum/test_sinc_bin_near_far.py tests/energy-grid/test_line_sparse_near_286.py \
  || echo "FAILED cuda tests rc=$?"
script=checks/cxr_bin_mean_profile_1003.py
uv run --no-sync python "$script" capture --grid "$out/grid-hbn-seed0.npy" \
  --out "$out/hbn5mev-cuda-n128.npz" --electrons 128 --core cuda
for fixture in cuda cpu; do
  uv run --no-sync python "$script" stats "$out/hbn5mev-$fixture-n128.npz"
  uv run --no-sync python "$script" replay "$out/hbn5mev-$fixture-n128.npz" \
    --out "$out/replay-$fixture.json"
done
if command -v ncu >/dev/null; then
  ncu --section SpeedOfLight --section Occupancy --section ComputeWorkloadAnalysis \
    --section WarpStateStats --section LaunchStats \
    --kernel-name regex:'pyrite_bin_tree_reduce|pyrite_sincsq_bin_mean_reduce|sparse_near' \
    --launch-count 12 \
    uv run --no-sync python "$script" replay "$out/hbn5mev-cuda-n128.npz" \
      --out "$out/replay-ncu.json" --slice-lines 50000 --max-flushes 1 --repeats 1 \
      --arms auto,pairs,warp,sparse,warp+series,sparse+series \
    > "$out/ncu.txt" 2>&1 || echo "FAILED ncu rc=$?"
else
  echo "ncu not found"
fi
echo DONE
